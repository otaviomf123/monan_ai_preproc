#!/usr/bin/env python
"""
Script COMPLEMENTAR para baixar ozônio (ozone_mass_mixing_ratio) em 17 níveis.

Você já tem O3 em 3 níveis estratosféricos (10, 30, 50 hPa) no script original.
Este script baixa O3 nos outros 14 níveis para completar os 17 níveis uniformes:

Níveis completos (17): 10, 30, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
                       750, 850, 925, 950, 1000 hPa

Níveis já baixados (3): 10, 30, 50 (no grupo 'stratosphere' do script original)
Níveis a baixar (14): 100, 150, 200, 250, 300, 400, 500, 600, 700, 750, 850,
                      925, 950, 1000

Arquivos de saída: data_ozone_extended/YYYY/era5_pressure_ozone_extended_YYYY_MM.nc

No pós-processamento, concatenar com o O3 estratosférico do script original
no eixo de pressure_level para formar perfil completo de 17 níveis.

Uso:
    python era5_download_ozone_extended.py
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

OUTPUT_ROOT = "data_ozone_extended"

# Níveis de O3 a baixar (complementam os 3 estratosféricos já existentes)
# Total uniforme final: 17 níveis (10, 30, 50 já existem; +14 aqui)
OZONE_LEVELS_TO_DOWNLOAD = [
    '100', '150', '200', '250', '300',        # UTLS + upper troposphere
    '400', '500', '600', '700', '750',        # mid/lower troposphere
    '850', '925', '950', '1000'               # boundary layer + surface
]

OZONE_VARIABLE = ['ozone_mass_mixing_ratio']

HOURS = ['00:00', '06:00', '12:00', '18:00']

START_YEAR = 1950
END_YEAR = 2025

MAX_PARALLEL_DOWNLOADS = 5
MAX_REQUESTS_PER_MINUTE = 20


def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(
        log_dir,
        f"era5_ozone_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            logger.warning(f"Arquivo inválido: {os.path.basename(filepath)} - {str(e)[:100]}")
        return False


def validate_and_clean_directory(year_dir, logger=None):
    import warnings
    warnings.filterwarnings('ignore')

    corrupted_files = []
    nc_files = glob.glob(os.path.join(year_dir, "*.nc"))

    if not nc_files:
        return corrupted_files

    if logger:
        logger.info(f"Verificando {len(nc_files)} arquivos em {year_dir}...")

    for filepath in nc_files:
        if not validate_file(filepath, logger):
            corrupted_files.append(filepath)
            try:
                os.remove(filepath)
                if logger:
                    logger.info(f"  Removido: {os.path.basename(filepath)}")
            except Exception as e:
                if logger:
                    logger.error(f"  Falha ao remover: {e}")

    return corrupted_files


def check_missing_files(year):
    year_dir = os.path.join(OUTPUT_ROOT, str(year))
    missing_months = []

    if not os.path.exists(year_dir):
        return list(range(1, 13))

    for month in range(1, 13):
        filename = f'era5_pressure_ozone_extended_{year}_{month:02d}.nc'
        filepath = os.path.join(year_dir, filename)
        if not os.path.exists(filepath):
            missing_months.append(month)

    return missing_months


class ERA5OzoneExtendedDownloader:
    def __init__(self, max_parallel=5, max_rpm=20):
        self.client = cdsapi.Client()
        self.logger = setup_logging()
        self.semaphore = Semaphore(max_parallel)
        self.max_parallel = max_parallel
        self.max_rpm = max_rpm
        self.request_times = []

        self.status_dir = "download_status_ozone"
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
                    if time.time() - start_time > 3600:
                        self.logger.warning(f"Download travado: {description}")
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
                    raise Exception(f"Arquivo muito pequeno: {temp_file}")

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

    def prepare_request(self, year, month):
        year_dir = os.path.join(OUTPUT_ROOT, str(year))
        os.makedirs(year_dir, exist_ok=True)

        month_str = f'{month:02d}'
        days = get_days(year, month)

        output_file = os.path.join(
            year_dir,
            f'era5_pressure_ozone_extended_{year}_{month_str}.nc'
        )

        request = {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': OZONE_VARIABLE,
            'pressure_level': OZONE_LEVELS_TO_DOWNLOAD,
            'year': str(year),
            'month': month_str,
            'day': days,
            'time': HOURS,
            'grid': ['0.25', '0.25'],
        }

        return {
            'output_file': output_file,
            'request': request,
            'description': f"O3 extended {year}-{month_str}"
        }

    def download_year(self, year):
        year_dir = os.path.join(OUTPUT_ROOT, str(year))

        if os.path.exists(year_dir):
            validate_and_clean_directory(year_dir, self.logger)

        missing_months = check_missing_files(year)

        if not missing_months:
            self.logger.info(f"Ano {year}: todos os meses existem!")
            return {'completed': 0, 'failed': 0, 'skipped': 0}

        self.logger.info(f"Ano {year}: {len(missing_months)} meses faltantes")

        requests = [self.prepare_request(year, m) for m in missing_months]

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
    downloader = ERA5OzoneExtendedDownloader(
        max_parallel=MAX_PARALLEL_DOWNLOADS,
        max_rpm=MAX_REQUESTS_PER_MINUTE
    )

    downloader.logger.info("=" * 70)
    downloader.logger.info("ERA5 OZONE EXTENDED PROFILE DOWNLOADER")
    downloader.logger.info("=" * 70)
    downloader.logger.info(f"Período: {START_YEAR}-{END_YEAR}")
    downloader.logger.info(f"Níveis: {len(OZONE_LEVELS_TO_DOWNLOAD)} (complementares a 10/30/50 hPa)")
    downloader.logger.info(f"Níveis baixados aqui: {OZONE_LEVELS_TO_DOWNLOAD}")
    downloader.logger.info(f"Resolução temporal: 6-horária")
    downloader.logger.info(f"Resolução espacial: 1.0°")
    downloader.logger.info(f"Diretório de saída: {OUTPUT_ROOT}/")
    downloader.logger.info("=" * 70)
    downloader.logger.info("")
    downloader.logger.info("OBSERVAÇÃO:")
    downloader.logger.info("Concatenar com O3 estratosférico (10/30/50 hPa) dos")
    downloader.logger.info("arquivos originais era5_pressure_stratosphere_YYYY_MM.nc")
    downloader.logger.info("no eixo 'pressure_level' em pós-processamento.")
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

        time.sleep(2)

    downloader.logger.info("\n" + "=" * 70)
    downloader.logger.info("DOWNLOAD OZÔNIO COMPLETO!")
    downloader.logger.info("=" * 70)
    downloader.logger.info(f"Arquivos baixados: {total_stats['completed']}")
    downloader.logger.info(f"Já existentes: {total_stats['skipped']}")
    downloader.logger.info(f"Falhas: {total_stats['failed']}")
    downloader.logger.info("=" * 70)


if __name__ == "__main__":
    main()
