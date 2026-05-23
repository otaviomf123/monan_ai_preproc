#!/usr/bin/env python
"""
Script para download EXCLUSIVO de variáveis ERA5 HORÁRIAS (acumuladas)
Baixa apenas os grupos que requerem dados 00-23 UTC para acumulação diária correta:
- precipitation (total_precipitation)
- evaporation (evaporation, potential_evaporation)
- heat_flux (surface_sensible_heat_flux, surface_latent_heat_flux)
- radiation_solar (5 variáveis de radiação solar)
- radiation_thermal (4 variáveis de radiação térmica)
- runoff (surface_runoff, sub_surface_runoff, runoff)
- snow (snowfall, snowmelt)

NÃO BAIXA: temperature, wind, pressure, soil, clouds, etc (já no script principal)
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

def setup_logging():
    """Configura logging thread-safe"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"era5_hourly_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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
    """Retorna lista de dias do mês"""
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

def check_missing_hourly_files(year_dir):
    """Verifica quais arquivos horários estão faltando"""
    missing = []
    
    # APENAS os 7 grupos horários
    hourly_groups = [
        'precipitation',
        'evaporation', 
        'heat_flux',
        'radiation_solar',
        'radiation_thermal',
        'runoff',
        'snow'
    ]
    
    for month in range(1, 13):
        month_str = f'{month:02d}'
        for group in hourly_groups:
            filename = f'era5_surface_hourly_{group}_{os.path.basename(year_dir)}_{month_str}.nc'
            filepath = os.path.join(year_dir, filename)
            
            if not os.path.exists(filepath):
                missing.append({'group': group, 'month': month})
            elif not validate_file(filepath):
                # Remove arquivo corrompido
                try:
                    os.remove(filepath)
                    missing.append({'group': group, 'month': month})
                except:
                    pass
    
    return missing

class ERA5HourlyDownloader:
    def __init__(self, max_parallel=3, max_rpm=15):
        """
        Inicializa downloader para variáveis horárias
        Args:
            max_parallel: Downloads simultâneos (default: 3 - mais conservador)
            max_rpm: Requisições por minuto (default: 15 - mais conservador)
        """
        self.client = cdsapi.Client()
        self.logger = setup_logging()
        self.semaphore = Semaphore(max_parallel)
        self.max_parallel = max_parallel
        self.max_rpm = max_rpm
        self.request_times = []
        
        self.status_dir = "download_status_hourly"
        if not os.path.exists(self.status_dir):
            os.makedirs(self.status_dir)
    
    def rate_limit_check(self):
        """Aplica rate limiting"""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= self.max_rpm:
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                self.logger.info(f"Rate limit atingido. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    def download_single_request(self, request_info):
        """Executa uma requisição de download"""
        output_file = request_info['output_file']
        description = request_info['description']
        
        # Verifica se existe e é válido
        if os.path.exists(output_file) and validate_file(output_file, self.logger):
            self.logger.debug(f"Já existe e válido: {output_file}")
            return {'status': 'skipped', 'file': output_file}
        
        status_file = os.path.join(self.status_dir, f"{os.path.basename(output_file)}.status")
        
        # Verifica se está em andamento
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
                if status.get('status') == 'downloading':
                    start_time = status.get('start_time', 0)
                    if time.time() - start_time > 3600:  # Travado >1h
                        self.logger.warning(f"Download travado: {description}")
                        os.remove(status_file)
                    else:
                        return {'status': 'in_progress', 'file': output_file}
        
        with self.semaphore:
            try:
                # Marca como downloading
                with open(status_file, 'w') as f:
                    json.dump({'status': 'downloading', 'start_time': time.time()}, f)
                
                self.rate_limit_check()
                self.logger.info(f"Baixando: {description}")
                
                temp_file = output_file + '.tmp'
                
                # Download
                self.client.retrieve(
                    'reanalysis-era5-single-levels',
                    request_info['request'],
                    temp_file
                )
                
                # Validação
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                    # Verifica se é ZIP
                    import zipfile
                    if zipfile.is_zipfile(temp_file):
                        self.logger.info(f"Extraindo ZIP...")
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
                    
                    # Valida integridade
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
                with open(status_file, 'w') as f:
                    json.dump({'status': 'error', 'error': str(e), 'time': time.time()}, f)
                
                self.logger.error(f"✗ Erro em {description}: {str(e)}")
                
                # Remove parciais
                for f in [output_file, temp_file, output_file + '.tmp']:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except:
                            pass
                
                return {'status': 'error', 'file': output_file, 'error': str(e)}
    
    def prepare_hourly_requests(self, year, month):
        """Prepara requisições APENAS para grupos horários"""
        year_dir = str(year)
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)
        
        month_str = f'{month:02d}'
        days = get_days(year, month)
        
        # TODAS as 24 horas do dia (00-23 UTC)
        hours = [f'{h:02d}:00' for h in range(24)]
        
        # APENAS GRUPOS ACUMULADOS (requer dados horários)
        hourly_groups = {
            'precipitation': [
                'total_precipitation',  # m/hora → converter para mm/dia
            ],
            'evaporation': [
                'evaporation',           # m/hora → converter para mm/dia
                'potential_evaporation', # m/hora → converter para mm/dia
            ],
            'heat_flux': [
                'surface_sensible_heat_flux',  # J/m²/hora → W/m² (média diária)
                'surface_latent_heat_flux',    # J/m²/hora → W/m² (média diária)
            ],
            'radiation_solar': [
                'surface_net_solar_radiation',              # J/m²
                'surface_solar_radiation_downwards',        # J/m²
                'top_net_solar_radiation',                  # J/m²
                'toa_incident_solar_radiation',             # J/m²
                'surface_solar_radiation_downward_clear_sky',  # J/m²
            ],
            'radiation_thermal': [
                'surface_net_thermal_radiation',              # J/m²
                'surface_thermal_radiation_downwards',        # J/m²
                'top_net_thermal_radiation',                  # J/m²
                'surface_thermal_radiation_downward_clear_sky',  # J/m²
            ],
            'runoff': [
                'surface_runoff',      # m/hora → mm/dia
                'sub_surface_runoff',  # m/hora → mm/dia
                'runoff',              # m/hora → mm/dia
            ],
            'snow': [
                'snowfall',   # m of water equivalent/hora
                'snowmelt',   # m of water equivalent/hora
            ],
        }
        
        requests = []
        for group_name, variables in hourly_groups.items():
            output_file = os.path.join(
                year_dir, 
                f'era5_surface_hourly_{group_name}_{year}_{month_str}.nc'
            )
            
            request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': variables,
                'year': str(year),
                'month': month_str,
                'day': days,
                'time': hours,  # TODAS as 24 horas
                'grid': ['0.25', '0.25'],
            }
            
            requests.append({
                'type': 'surface',
                'output_file': output_file,
                'request': request,
                'description': f"Hourly {group_name} {year}-{month_str}"
            })
        
        return requests
    
    def download_year(self, year, start_month=1, end_month=12):
        """Baixa dados horários de um ano"""
        year_dir = str(year)
        
        # Verifica arquivos faltantes
        if os.path.exists(year_dir):
            missing = check_missing_hourly_files(year_dir)
            if not missing:
                self.logger.info(f"Ano {year}: todos os arquivos horários já completos!")
                return {'completed': 0, 'failed': 0, 'skipped': 0}
            
            self.logger.info(f"Ano {year}: {len(missing)} arquivos faltantes")
        
        total_stats = {'completed': 0, 'failed': 0, 'skipped': 0}
        
        for month in range(start_month, end_month + 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processando {year}/{month:02d}")
            self.logger.info(f"{'='*50}")
            
            requests = self.prepare_hourly_requests(year, month)
            stats = self.execute_downloads(requests)
            
            total_stats['completed'] += stats['completed']
            total_stats['failed'] += stats['failed']
            total_stats['skipped'] += stats['skipped']
            
            time.sleep(2)  # Pausa entre meses
        
        return total_stats
    
    def execute_downloads(self, requests):
        """Executa downloads em paralelo"""
        total = len(requests)
        self.logger.info(f"Requisições preparadas: {total}")
        
        completed = failed = skipped = 0
        
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_req = {
                executor.submit(self.download_single_request, req): req 
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
    """Função principal"""
    
    # CONFIGURAÇÕES
    MAX_PARALLEL = 15   # Conservador para downloads horários (grandes)
    MAX_RPM = 15       # Limite mais conservador
    
    START_YEAR = 1950
    END_YEAR = 2025
    
    downloader = ERA5HourlyDownloader(
        max_parallel=MAX_PARALLEL,
        max_rpm=MAX_RPM
    )
    
    downloader.logger.info("=" * 70)
    downloader.logger.info("ERA5 HOURLY ACCUMULATED VARIABLES DOWNLOADER")
    downloader.logger.info("=" * 70)
    downloader.logger.info(f"Período: {START_YEAR}-{END_YEAR}")
    downloader.logger.info("Resolução temporal: 1 HORA (00-23 UTC)")
    downloader.logger.info("Resolução espacial: 1.0°")
    downloader.logger.info("Formato: NetCDF")
    downloader.logger.info(f"Downloads paralelos: {MAX_PARALLEL}")
    downloader.logger.info("")
    downloader.logger.info("GRUPOS BAIXADOS (apenas acumulados):")
    downloader.logger.info("  1. precipitation (1 var)")
    downloader.logger.info("  2. evaporation (2 vars)")
    downloader.logger.info("  3. heat_flux (2 vars)")
    downloader.logger.info("  4. radiation_solar (5 vars)")
    downloader.logger.info("  5. radiation_thermal (4 vars)")
    downloader.logger.info("  6. runoff (3 vars)")
    downloader.logger.info("  7. snow (2 vars)")
    downloader.logger.info("=" * 70)
    
    total_stats = {'completed': 0, 'failed': 0, 'skipped': 0}
    
    for year in range(START_YEAR, END_YEAR + 1):
        downloader.logger.info(f"\n{'#'*50}")
        downloader.logger.info(f"ANO: {year}")
        downloader.logger.info(f"{'#'*50}")
        
        # Para 2025, apenas até agosto
        end_month = 8 if year == 2025 else 12
        
        try:
            stats = downloader.download_year(year, end_month=end_month)
            
            total_stats['completed'] += stats['completed']
            total_stats['failed'] += stats['failed']
            total_stats['skipped'] += stats['skipped']
            
            downloader.logger.info(f"Ano {year}: {stats}")
            
        except Exception as e:
            downloader.logger.error(f"Erro no ano {year}: {e}")
        
        time.sleep(3)  # Pausa entre anos
    
    # RELATÓRIO FINAL
    downloader.logger.info("\n" + "=" * 70)
    downloader.logger.info("DOWNLOAD HORÁRIO COMPLETO!")
    downloader.logger.info("=" * 70)
    downloader.logger.info(f"Arquivos baixados: {total_stats['completed']}")
    downloader.logger.info(f"Já existentes: {total_stats['skipped']}")
    downloader.logger.info(f"Falhas: {total_stats['failed']}")
    downloader.logger.info("=" * 70)
    
    downloader.logger.info("\nPÓS-PROCESSAMENTO NECESSÁRIO:")
    downloader.logger.info("Os arquivos horários precisam ser agregados para diários:")
    downloader.logger.info("")
    downloader.logger.info("import xarray as xr")
    downloader.logger.info("# Abrir arquivo horário")
    downloader.logger.info("ds = xr.open_dataset('1950/era5_surface_hourly_precipitation_1950_01.nc')")
    downloader.logger.info("")
    downloader.logger.info("# Converter m → mm")
    downloader.logger.info("ds['total_precipitation'] = ds['total_precipitation'] * 1000")
    downloader.logger.info("")
    downloader.logger.info("# Acumular para diário")
    downloader.logger.info("daily = ds.resample(time='1D').sum()")
    downloader.logger.info("")
    downloader.logger.info("# Para radiação (J/m²) → média W/m²")
    downloader.logger.info("ds_rad = xr.open_dataset('.../radiation_solar_...')")
    downloader.logger.info("daily_rad = ds_rad.resample(time='1D').sum() / 86400")
    downloader.logger.info("")
    downloader.logger.info("IMPORTANTE: total_precipitation JÁ é o total")
    downloader.logger.info("NÃO somar convective + large_scale (isso é o detalhamento)")

if __name__ == "__main__":
    main()
