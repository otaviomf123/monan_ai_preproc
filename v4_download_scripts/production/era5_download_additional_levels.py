#!/usr/bin/env python
"""
Script COMPLEMENTAR para baixar níveis adicionais (250, 950 hPa)
das variáveis core ERA5 que já foram baixadas nos 15 níveis originais.

Níveis adicionais:
- 250 hPa: resolve tropopausa extratropical, jet stream, cirrus altos
- 950 hPa: resolve camada limite / nuvens baixas

Variáveis baixadas nesses níveis (mesmas do script original):
- geopotential
- temperature
- u_component_of_wind
- v_component_of_wind
- relative_humidity

Arquivos de saída: data_additional_levels/YYYY/era5_pressure_additional_levels_YYYY_MM.nc
No pós-processamento, estes arquivos devem ser concatenados ao longo do eixo
de pressure_level com os arquivos originais (15 níveis) para formar os 17 níveis
consolidados.

Uso:
    python era5_download_additional_levels.py
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

# Diretório de saída separado para não misturar com os dados originais
OUTPUT_ROOT = "data_additional_levels"

# Níveis adicionais (apenas esses dois)
ADDITIONAL_LEVELS = ['250', '950']

# Variáveis core (mesmas do script original)
CORE_VARIABLES = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'relative_humidity',
]

# Horas (6-horário, igual ao script original)
HOURS = ['00:00', '06:00', '12:00', '18:00']

# Período
START_YEAR = 1950
END_YEAR = 2025

# Paralelismo
MAX_PARALLEL_DOWNLOADS = 5
MAX_REQUESTS_PER_MINUTE = 20


def setup_logging():
    """Configura logging thread-safe"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(
        log_dir,
        f"era5_additional_levels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    """Retorna lista de dias para o mês especificado"""
    _, num_days = calendar.monthrange(year, month)
    return [f'{d:02d}' for d in range(1, num_days + 1)]


def validate_file(filepath, logger=None):
    """Valida se arquivo NetCDF é válido"""
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
    """Remove arquivos NetCDF corrompidos em um diretório"""
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
                    logger.info(f"  Removido corrompido: {os.path.basename(filepath)}")
            except Exception as e:
                if logger:
                    logger.error(f"  Falha ao remover: {e}")

    return corrupted_files


def check_missing_files(year):
    """Verifica quais meses estão faltando para um ano"""
    year_dir = os.path.join(OUTPUT_ROOT, str(year))
    missing_months = []

    if not os.path.exists(year_dir):
        return list(range(1, 13))

    for month in range(1, 13):
        filename = f'era5_pressure_additional_levels_{year}_{month:02d}.nc'
        filepath = os.path.join(year_dir, filename)
        if not os.path.exists(filepath):
            missing_months.append(month)

    return missing_months


class ERA5AdditionalLevelsDownloader:
    def __init__(self, max_parallel=5, max_rpm=20):
        self.client = cdsapi.Client()
        self.logger = setup_logging()
        self.semaphore = Semaphore(max_parallel)
        self.max_parallel = max_parallel
        self.max_rpm = max_rpm
        self.request_times = []

        self.status_dir = "download_status_additional"
        if not os.path.exists(self.status_dir):
            os.makedirs(self.status_dir)

        if not os.path.exists(OUTPUT_ROOT):
            os.makedirs(OUTPUT_ROOT)

    def rate_limit_check(self):
        """Rate limiting"""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        if len(self.request_times) >= self.max_rpm:
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                self.logger.info(f"Rate limit atingido. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        self.request_times.append(current_time)

    def download_single(self, request_info):
        """Executa uma requisição de download"""
        output_file = request_info['output_file']
        description = request_info['description']

        # Pula se já existe e é válido
        if os.path.exists(output_file) and validate_file(output_file, self.logger):
            self.logger.debug(f"Já existe: {output_file}")
            return {'status': 'skipped', 'file': output_file}

        status_file = os.path.join(
            self.status_dir,
            f"{os.path.basename(output_file)}.status"
        )

        # Verifica download em andamento
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

                # Validação
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                    # Verifica se é ZIP
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
                        raise Exception("Arquivo corrompido após download")
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
        """Prepara requisição para os níveis adicionais"""
        year_dir = os.path.join(OUTPUT_ROOT, str(year))
        os.makedirs(year_dir, exist_ok=True)

        month_str = f'{month:02d}'
        days = get_days(year, month)

        output_file = os.path.join(
            year_dir,
            f'era5_pressure_additional_levels_{year}_{month_str}.nc'
        )

        request = {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',
            'variable': CORE_VARIABLES,
            'pressure_level': ADDITIONAL_LEVELS,
            'year': str(year),
            'month': month_str,
            'day': days,
            'time': HOURS,
            'grid': ['1.0', '1.0'],
        }

        return {
            'output_file': output_file,
            'request': request,
            'description': f"Additional levels {year}-{month_str}"
        }

    def download_year(self, year):
        """Baixa todos os meses faltantes de um ano"""
        year_dir = os.path.join(OUTPUT_ROOT, str(year))

        # Valida arquivos existentes
        if os.path.exists(year_dir):
            validate_and_clean_directory(year_dir, self.logger)

        missing_months = check_missing_files(year)

        if not missing_months:
            self.logger.info(f"Ano {year}: todos os meses já existem!")
            return {'completed': 0, 'failed': 0, 'skipped': 0}

        self.logger.info(f"Ano {year}: {len(missing_months)} meses faltantes")

        requests = [self.prepare_request(year, m) for m in missing_months]

        return self.execute_downloads(requests)

    def execute_downloads(self, requests):
        """Executa downloads em paralelo"""
        total = len(requests)
        self.logger.info(f"Requisições preparadas: {total}")

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
    downloader = ERA5AdditionalLevelsDownloader(
        max_parallel=MAX_PARALLEL_DOWNLOADS,
        max_rpm=MAX_REQUESTS_PER_MINUTE
    )

    downloader.logger.info("=" * 70)
    downloader.logger.info("ERA5 ADDITIONAL PRESSURE LEVELS DOWNLOADER")
    downloader.logger.info("=" * 70)
    downloader.logger.info(f"Período: {START_YEAR}-{END_YEAR}")
    downloader.logger.info(f"Níveis adicionais: {ADDITIONAL_LEVELS} hPa")
    downloader.logger.info(f"Variáveis: {len(CORE_VARIABLES)} (core)")
    downloader.logger.info(f"Resolução temporal: 6-horária")
    downloader.logger.info(f"Resolução espacial: 1.0°")
    downloader.logger.info(f"Diretório de saída: {OUTPUT_ROOT}/")
    downloader.logger.info("=" * 70)
    downloader.logger.info("")
    downloader.logger.info("OBSERVAÇÃO:")
    downloader.logger.info("Estes arquivos devem ser concatenados ao longo do")
    downloader.logger.info("eixo 'pressure_level' com os arquivos originais")
    downloader.logger.info("(15 níveis) em pós-processamento para formar 17 níveis.")
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

    # Relatório final
    downloader.logger.info("\n" + "=" * 70)
    downloader.logger.info("DOWNLOAD COMPLETO!")
    downloader.logger.info("=" * 70)
    downloader.logger.info(f"Arquivos baixados: {total_stats['completed']}")
    downloader.logger.info(f"Já existentes: {total_stats['skipped']}")
    downloader.logger.info(f"Falhas: {total_stats['failed']}")
    downloader.logger.info("=" * 70)


if __name__ == "__main__":
    main()
