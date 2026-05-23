#!/usr/bin/env python
"""
Script COMPLEMENTAR para baixar hidrometeoros em perfil vertical + cloud fraction.

Variáveis (5 no total, essenciais para all-sky DA com pyCRTM/RTTOV):
- specific_cloud_liquid_water_content (CLWC) [kg/kg]
- specific_cloud_ice_water_content (CIWC) [kg/kg]
- specific_rain_water_content (CRWC) [kg/kg]
- specific_snow_water_content (CSWC) [kg/kg]
- fraction_of_cloud_cover (CC) [0-1]

Níveis: 17 níveis uniformes (mesmos do resto do pipeline)
        10, 30, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
        750, 850, 925, 950, 1000 hPa

Nota: em níveis estratosféricos (10, 30, 50 hPa) hidrometeoros são
essencialmente zero. Baixamos mesmo assim para manter uniformidade vertical
com o resto do dataset. Storage adicional é pequeno.

Organização dos grupos (respeitar limites do CDS):
- Grupo 1: CLWC + CIWC em todos os 17 níveis (nuvens não-precipitantes)
- Grupo 2: CRWC + CSWC em todos os 17 níveis (hidrometeoros precipitantes)
- Grupo 3: CC em todos os 17 níveis (fraction de cobertura)

Cada grupo é um arquivo separado para facilitar paralelização e evitar
timeouts em requests grandes.

Arquivos de saída:
    data_hydrometeors/YYYY/era5_pressure_hydro_nonprecip_YYYY_MM.nc
    data_hydrometeors/YYYY/era5_pressure_hydro_precip_YYYY_MM.nc
    data_hydrometeors/YYYY/era5_pressure_cloud_fraction_YYYY_MM.nc

Uso:
    python era5_download_hydrometeors.py
"""

import cdsapi
import calendar
import os
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import json
import glob

# ============================================================
# CONFIGURAÇÃO
# ============================================================

OUTPUT_ROOT = "data_hydrometeors"

# 17 níveis uniformes (inclui 250 e 950 adicionais para consistência)
PRESSURE_LEVELS_17 = [
    '10', '30', '50',
    '100', '150', '200', '250', '300',
    '400', '500', '600', '700', '750',
    '850', '925', '950', '1000'
]

# Grupos de variáveis (separados para evitar requests enormes no CDS)
HYDROMETEOR_GROUPS = {
    'hydro_nonprecip': [
        'specific_cloud_liquid_water_content',
        'specific_cloud_ice_water_content',
    ],
    'hydro_precip': [
        'specific_rain_water_content',
        'specific_snow_water_content',
    ],
    'cloud_fraction': [
        'fraction_of_cloud_cover',
    ],
}

HOURS = ['00:00', '06:00', '12:00', '18:00']

START_YEAR = 1950
END_YEAR = 2025

# Hidrometeoros são pesados; paralelismo mais conservador
MAX_PARALLEL_DOWNLOADS = 4
MAX_REQUESTS_PER_MINUTE = 15


def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(
        log_dir,
        f"era5_hydrometeors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(threadName)-10s] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def get_days(year, month):
    _, num_days = calendar.monthrange(year, month)
    return [f'{d:02d}' for d in range(1, num_days + 1)]


def validate_file(filepath, logger=None):
    try:
        import xarray as xr
        import warnings
        warnings.filterwarnings('ignore')

        with xr.open_dataset(filepath):
            pass
        return True
    except Exception as e:
        if logger:
            logger.warning(f"Inválido: {os.path.basename(filepath)} - {str(e)[:100]}")
        return False


def validate_and_clean_directory(year_dir, logger=None):
    import warnings
    warnings.filterwarnings('ignore')

    corrupted_files = []
    nc_files = glob.glob(os.path.join(year_dir, "*.nc"))

    if not nc_files:
        return corrupted_files

    if logger:
        logger.info(f"Verificando {len(nc_files)} em {year_dir}...")

    for filepath in nc_files:
        if not validate_file(filepath, logger):
            corrupted_files.append(filepath)
            try:
                os.remove(filepath)
                if logger:
                    logger.info(f"  Removido: {os.path.basename(filepath)}")
            except Exception as e:
                if logger:
                    logger.error(f"  Falha: {e}")

    return corrupted_files


def check_missing_files(year):
    """Verifica quais (grupo, mês) estão faltando para um ano"""
    year_dir = os.path.join(OUTPUT_ROOT, str(year))
    missing = []

    for month in range(1, 13):
        month_str = f'{month:02d}'
        for group_name in HYDROMETEOR_GROUPS.keys():
            filename = f'era5_pressure_{group_name}_{year}_{month_str}.nc'
            filepath = os.path.join(year_dir, filename)
            if not os.path.exists(filepath):
                missing.append({'group': group_name, 'month': month})

    return missing


class ERA5HydrometeorsDownloader:
    def __init__(self, max_parallel=4, max_rpm=15):
        self.client = cdsapi.Client()
        self.logger = setup_logging()
        self.semaphore = Semaphore(max_parallel)
        self.max_parallel = max_parallel
        self.max_rpm = max_rpm
        self.request_times = []

        self.status_dir = "download_status_hydrometeors"
        if not os.path.exists(self.status_dir):
            os.makedirs(self.status_dir)

        if not os.path.exists(OUTPUT_ROOT):
            os.makedirs(OUTPUT_ROOT)

    def rate_limit_check(self):
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        if len(self.request_times) >= self.max_rpm:
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                self.logger.info(f"Rate limit. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        self.request_times.append(current_time)

    def download_single(self, request_info):
        output_file = request_info['output_file']
        description = request_info['description']

        if os.path.exists(output_file) and validate_file(output_file, self.logger):
            self.logger.debug(f"Já existe: {output_file}")
            return {'status': 'skipped', 'file': output_file}

        status_file = os.path.join(
            self.status_dir,
            f"{os.path.basename(output_file)}.status"
        )

        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
                if status.get('status') == 'downloading':
                    start_time = status.get('start_time', 0)
                    # Hidrometeoros demoram mais; timeout maior (2h)
                    if time.time() - start_time > 7200:
                        self.logger.warning(f"Travado: {description}")
                        os.remove(status_file)
                    else:
                        return {'status': 'in_progress', 'file': output_file}

        with self.semaphore:
            try:
                with open(status_file, 'w') as f:
                    json.dump({'status': 'downloading', 'start_time': time.time()}, f)

                self.rate_limit_check()
                self.logger.info(f"Baixando: {description}")

                temp_file = output_file + '.tmp'

                self.client.retrieve(
                    'reanalysis-era5-pressure-levels',
                    request_info['request'],
                    temp_file
                )

                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                    import zipfile
                    if zipfile.is_zipfile(temp_file):
                        self.logger.info("Extraindo ZIP...")
                        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                            nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                            if nc_files:
                                extracted = temp_file + '.nc'
                                with open(extracted, 'wb') as f:
                                    f.write(zip_ref.read(nc_files[0]))
                                os.remove(temp_file)
                                temp_file = extracted
                            else:
                                raise Exception("Nenhum .nc no ZIP")

                    if validate_file(temp_file, self.logger):
                        os.rename(temp_file, output_file)
                        self.logger.info(f"✓ Completo: {description}")

                        with open(status_file, 'w') as f:
                            json.dump({'status': 'completed', 'end_time': time.time()}, f)

                        return {'status': 'success', 'file': output_file}
                    else:
                        raise Exception("Arquivo corrompido")
                else:
                    raise Exception(f"Muito pequeno: {temp_file}")

            except Exception as e:
                self.logger.error(f"✗ Erro: {description} - {str(e)}")

                for f in [temp_file, temp_file + '.nc', output_file + '.tmp']:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except:
                            pass

                with open(status_file, 'w') as f:
                    json.dump({'status': 'error', 'error': str(e), 'time': time.time()}, f)

                return {'status': 'error', 'file': output_file, 'error': str(e)}

    def prepare_request(self, year, month, group_name, variables):
        year_dir = os.path.join(OUTPUT_ROOT, str(year))
        os.makedirs(year_dir, exist_ok=True)

        month_str = f'{month:02d}'
        days = get_days(year, month)

        output_file = os.path.join(
            year_dir,
            f'era5_pressure_{group_name}_{year}_{month_str}.nc'
        )

        request = {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': variables,
            'pressure_level': PRESSURE_LEVELS_17,
            'year': str(year),
            'month': month_str,
            'day': days,
            'time': HOURS,
            'grid': ['1.0', '1.0'],
        }

        return {
            'output_file': output_file,
            'request': request,
            'description': f"{group_name} {year}-{month_str}"
        }

    def download_year(self, year):
        year_dir = os.path.join(OUTPUT_ROOT, str(year))

        if os.path.exists(year_dir):
            validate_and_clean_directory(year_dir, self.logger)

        missing_items = check_missing_files(year)

        if not missing_items:
            self.logger.info(f"Ano {year}: completo!")
            return {'completed': 0, 'failed': 0, 'skipped': 0}

        self.logger.info(f"Ano {year}: {len(missing_items)} arquivos faltantes")

        requests = []
        for item in missing_items:
            group_name = item['group']
            month = item['month']
            variables = HYDROMETEOR_GROUPS[group_name]

            req = self.prepare_request(year, month, group_name, variables)
            requests.append(req)

        return self.execute_downloads(requests)

    def execute_downloads(self, requests):
        total = len(requests)
        self.logger.info(f"Requisições: {total}")

        completed = failed = skipped = 0

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_req = {
                executor.submit(self.download_single, req): req
                for req in requests
            }

            for future in as_completed(future_to_req):
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        completed += 1
                    elif result['status'] == 'error':
                        failed += 1
                    else:
                        skipped += 1

                    done = completed + failed + skipped
                    self.logger.info(
                        f"Progresso: {done}/{total} "
                        f"(✓{completed} | ✗{failed} | ⊘{skipped})"
                    )
                except Exception as e:
                    self.logger.error(f"Erro no executor: {e}")
                    failed += 1

        return {'completed': completed, 'failed': failed, 'skipped': skipped}


def main():
    downloader = ERA5HydrometeorsDownloader(
        max_parallel=MAX_PARALLEL_DOWNLOADS,
        max_rpm=MAX_REQUESTS_PER_MINUTE
    )

    downloader.logger.info("=" * 70)
    downloader.logger.info("ERA5 HYDROMETEORS + CLOUD FRACTION DOWNLOADER")
    downloader.logger.info("=" * 70)
    downloader.logger.info(f"Período: {START_YEAR}-{END_YEAR}")
    downloader.logger.info(f"Níveis: {len(PRESSURE_LEVELS_17)} (uniformes com resto do pipeline)")
    downloader.logger.info(f"Variáveis totais: 5")
    downloader.logger.info(f"  - CLWC: specific_cloud_liquid_water_content")
    downloader.logger.info(f"  - CIWC: specific_cloud_ice_water_content")
    downloader.logger.info(f"  - CRWC: specific_rain_water_content")
    downloader.logger.info(f"  - CSWC: specific_snow_water_content")
    downloader.logger.info(f"  - CC:   fraction_of_cloud_cover")
    downloader.logger.info(f"Resolução temporal: 6-horária")
    downloader.logger.info(f"Resolução espacial: 1.0°")
    downloader.logger.info(f"Diretório de saída: {OUTPUT_ROOT}/")
    downloader.logger.info("=" * 70)
    downloader.logger.info("")
    downloader.logger.info("ORGANIZAÇÃO POR MÊS (3 arquivos):")
    downloader.logger.info("  1. era5_pressure_hydro_nonprecip_YYYY_MM.nc (CLWC + CIWC)")
    downloader.logger.info("  2. era5_pressure_hydro_precip_YYYY_MM.nc    (CRWC + CSWC)")
    downloader.logger.info("  3. era5_pressure_cloud_fraction_YYYY_MM.nc  (CC)")
    downloader.logger.info("=" * 70)
    downloader.logger.info("")
    downloader.logger.info("NOTAS IMPORTANTES:")
    downloader.logger.info("- Hidrometeoros são ESSENCIALMENTE ZERO em 10/30/50 hPa")
    downloader.logger.info("  mas são baixados para manter uniformidade vertical")
    downloader.logger.info("- No pré-processamento, aplicar clip + log-transform")
    downloader.logger.info("  antes de normalizar (heavy-tailed distributions)")
    downloader.logger.info("- Normalização recomendada: z-score POR-NÍVEL")
    downloader.logger.info("=" * 70)

    total_stats = {'completed': 0, 'failed': 0, 'skipped': 0}

    for year in range(START_YEAR, END_YEAR + 1):
        downloader.logger.info(f"\n{'#' * 50}")
        downloader.logger.info(f"ANO: {year}")
        downloader.logger.info(f"{'#' * 50}")

        try:
            stats = downloader.download_year(year)

            total_stats['completed'] += stats['completed']
            total_stats['failed'] += stats['failed']
            total_stats['skipped'] += stats['skipped']

            downloader.logger.info(f"Ano {year}: {stats}")

        except Exception as e:
            downloader.logger.error(f"Erro no ano {year}: {e}")

        time.sleep(3)

    downloader.logger.info("\n" + "=" * 70)
    downloader.logger.info("DOWNLOAD HIDROMETEOROS COMPLETO!")
    downloader.logger.info("=" * 70)
    downloader.logger.info(f"Arquivos baixados: {total_stats['completed']}")
    downloader.logger.info(f"Já existentes: {total_stats['skipped']}")
    downloader.logger.info(f"Falhas: {total_stats['failed']}")
    downloader.logger.info("=" * 70)


if __name__ == "__main__":
    main()
