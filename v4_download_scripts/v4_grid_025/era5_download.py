#!/usr/bin/env python
"""
Script para download de dados ERA5 a cada 6 horas (00, 06, 12, 18 UTC)
Versão atualizada com:
- Verificação de arquivos existentes antes de baixar
- Variáveis de neve não-acumuladas (snow depth, snow density)
- Variáveis de gelo (ice cover, temperature, thickness, type)
- Download apenas de arquivos faltantes
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
    
    log_file = os.path.join(log_dir, f"era5_6hourly_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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
    """Verifica e remove arquivos NetCDF corrompidos em um diretório"""
    import warnings
    warnings.filterwarnings('ignore')
    
    corrupted_files = []
    valid_files = []
    
    nc_files = glob.glob(os.path.join(year_dir, "*.nc"))
    
    if not nc_files:
        return corrupted_files
    
    if logger:
        logger.info(f"Verificando integridade de {len(nc_files)} arquivos em {year_dir}...")
    
    for filepath in nc_files:
        if validate_file(filepath, logger):
            valid_files.append(filepath)
        else:
            corrupted_files.append(filepath)
            if logger:
                logger.warning(f"Arquivo corrompido detectado: {os.path.basename(filepath)}")
            
            # Remove o arquivo corrompido
            try:
                os.remove(filepath)
                if logger:
                    logger.info(f"  Arquivo corrompido removido: {os.path.basename(filepath)}")
            except Exception as rm_error:
                if logger:
                    logger.error(f"  Não foi possível remover: {rm_error}")
    
    if logger:
        logger.info(f"Verificação completa: {len(valid_files)} válidos, {len(corrupted_files)} corrompidos")
    
    return corrupted_files

def check_missing_files(year_dir):
    """Verifica quais arquivos estão faltando em um diretório de ano existente"""
    missing_files = []
    
    # Grupos de superfície (6-horários)
    expected_surface_groups = [
        'temperature', 'temperature_extremes', 'sst',  # Temperatura
        'wind', 'pressure', 
        'precipitation_type',  # Tipo de precipitação (6h)
        'snow', 'snow_processes',  # Neve não-acumulada (snow depth, snow density)
        'soil_moisture', 'soil_temperature', 'clouds', 'water_column',
        'ocean_ice',  # Gelo marinho (cobertura, temperatura, espessura, tipo)
        'boundary_layer', 'vegetation'
    ]
    
    # Grupos de níveis de pressão
    expected_pressure_groups = [
        'stratosphere', 'upper_troposphere', 'mid_troposphere',
        'lower_troposphere', 'boundary_layer'
    ]
    
    # Verifica para cada mês do ano
    for month in range(1, 13):
        month_str = f'{month:02d}'
        
        # Verificar arquivos de superfície
        for group in expected_surface_groups:
            filename = f'era5_surface_{group}_{year_dir}_{month_str}.nc'
            filepath = os.path.join(year_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append({'type': 'surface', 'group': group, 'month': month})
        
        # Verificar arquivos de pressão
        for group in expected_pressure_groups:
            filename = f'era5_pressure_{group}_{year_dir}_{month_str}.nc'
            filepath = os.path.join(year_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append({'type': 'pressure', 'group': group, 'month': month})
    
    return missing_files

class ERA5Downloader6Hourly:
    def __init__(self, max_parallel_requests=5, max_requests_per_minute=20):
        """
        Inicializa o downloader com controle de concorrência
        
        Args:
            max_parallel_requests: Número máximo de downloads simultâneos
            max_requests_per_minute: Limite de requisições por minuto
        """
        self.client = cdsapi.Client()
        self.logger = setup_logging()
        self.semaphore = Semaphore(max_parallel_requests)
        self.max_parallel = max_parallel_requests
        self.max_rpm = max_requests_per_minute
        self.request_times = []
        
        # Diretório para status de downloads
        self.status_dir = "download_status"
        if not os.path.exists(self.status_dir):
            os.makedirs(self.status_dir)
    
    def rate_limit_check(self):
        """Verifica e aplica rate limiting"""
        current_time = time.time()
        
        # Remove requisições antigas (>1 minuto)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Se atingiu o limite, aguarda
        if len(self.request_times) >= self.max_rpm:
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                self.logger.info(f"Rate limit atingido. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    def download_single_request(self, request_info):
        """Executa uma única requisição de download"""
        output_file = request_info['output_file']
        description = request_info['description']
        
        # Verifica se arquivo já existe e é válido
        if os.path.exists(output_file) and validate_file(output_file, self.logger):
            self.logger.debug(f"Arquivo já existe e é válido: {output_file}")
            return {'status': 'skipped', 'file': output_file}
        
        # Status file para evitar downloads duplicados
        status_file = os.path.join(self.status_dir, f"{os.path.basename(output_file)}.status")
        
        # Verifica se outro processo está baixando
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
                if status.get('status') == 'downloading':
                    start_time = status.get('start_time', 0)
                    if time.time() - start_time > 3600:  # Download travado por >1h
                        self.logger.warning(f"Download travado detectado: {description}")
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
                
                # Faz o download
                self.client.retrieve(
                    'reanalysis-era5-single-levels' if request_info['type'] == 'surface' else 'reanalysis-era5-pressure-levels',
                    request_info['request'],
                    temp_file
                )
                
                # Validação básica do arquivo
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                    # Verifica se é um arquivo ZIP
                    import zipfile
                    if zipfile.is_zipfile(temp_file):
                        self.logger.info(f"Extraindo arquivo ZIP...")
                        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                            # Extrai apenas arquivos .nc
                            nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                            if nc_files:
                                extracted_file = temp_file + '.nc'
                                with open(extracted_file, 'wb') as f:
                                    f.write(zip_ref.read(nc_files[0]))
                                os.remove(temp_file)
                                temp_file = extracted_file
                            else:
                                raise Exception("Nenhum arquivo .nc encontrado no ZIP")
                    
                    # Valida integridade do NetCDF
                    if validate_file(temp_file, self.logger):
                        os.rename(temp_file, output_file)
                        self.logger.info(f"✓ Download completo: {description}")
                        
                        # Atualiza status
                        with open(status_file, 'w') as f:
                            json.dump({'status': 'completed', 'end_time': time.time()}, f)
                        
                        return {'status': 'success', 'file': output_file}
                    else:
                        raise Exception("Arquivo corrompido após download")
                else:
                    raise Exception(f"Arquivo muito pequeno ou não existe: {temp_file}")
                
            except Exception as e:
                self.logger.error(f"✗ Erro no download: {description}")
                self.logger.error(f"  Detalhes: {str(e)}")
                
                # Remove arquivos temporários
                for f in [temp_file, temp_file + '.nc', output_file + '.tmp']:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except:
                            pass
                
                # Atualiza status
                with open(status_file, 'w') as f:
                    json.dump({'status': 'error', 'error': str(e), 'time': time.time()}, f)
                
                return {'status': 'error', 'file': output_file, 'error': str(e)}
    
    def prepare_surface_requests(self, year, month):
        """Prepara requisições para dados de superfície (6-horários)"""
        year_dir = str(year)
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)
        
        month_str = f'{month:02d}'
        days = get_days(year, month)
        hours = ['00:00', '06:00', '12:00', '18:00']  # 6-horário
        
        # Dicionário de grupos de superfície
        surface_groups = {
            'temperature': [
                '2m_temperature',
                'skin_temperature',
                '2m_dewpoint_temperature',
            ],
            'temperature_extremes': [
                'maximum_2m_temperature_since_previous_post_processing',
                'minimum_2m_temperature_since_previous_post_processing',
            ],
            'sst': [
                'sea_surface_temperature',
            ],
            'wind': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                '10m_wind_gust_since_previous_post_processing',
            ],
            'pressure': [
                'surface_pressure',
                'mean_sea_level_pressure',
            ],
            'precipitation_type': [
                'precipitation_type',
            ],
            'snow': [
                'snow_depth',  # sd - profundidade da neve (instantâneo)
            ],
            'snow_processes': [
                'snow_density',  # rsn - densidade da neve (instantâneo)
            ],
            'soil_moisture': [
                'volumetric_soil_water_layer_1',
                'volumetric_soil_water_layer_2',
                'volumetric_soil_water_layer_3',
                'volumetric_soil_water_layer_4',
            ],
            'soil_temperature': [
                'soil_temperature_level_1',
                'soil_temperature_level_2',
                'soil_temperature_level_3',
                'soil_temperature_level_4',
            ],
            'clouds': [
                'total_cloud_cover',
                'low_cloud_cover',
                'medium_cloud_cover',
                'high_cloud_cover',
            ],
            'water_column': [
                'total_column_water',
                'total_column_water_vapour',
            ],
            'ocean_ice': [
                'sea_ice_cover',  # Cobertura de gelo (0-1)
                'ice_temperature_layer_1',  # Temperatura camada 1
                'ice_temperature_layer_2',  # Temperatura camada 2
                'ice_temperature_layer_3',  # Temperatura camada 3
                'ice_temperature_layer_4',  # Temperatura camada 4
                  #'sea_ice_thickness', Espessura do gelo ##                 'ice_type',  # Tipo de gelo
            ],
            'boundary_layer': [
                'boundary_layer_height',
                'boundary_layer_dissipation',
            ],
            'vegetation': [
                'leaf_area_index_low_vegetation',
                'leaf_area_index_high_vegetation',
                'low_vegetation_cover',
                'high_vegetation_cover',
                'type_of_low_vegetation',
                'type_of_high_vegetation',
            ],
        }
        
        requests = []
        for group_name, variables in surface_groups.items():
            output_file = os.path.join(
                year_dir, 
                f'era5_surface_{group_name}_{year}_{month_str}.nc'
            )
            
            request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': variables,
                'year': str(year),
                'month': month_str,
                'day': days,
                'time': hours,
                'grid': ['0.25', '0.25'],
            }
            
            requests.append({
                'type': 'surface',
                'output_file': output_file,
                'request': request,
                'description': f"Surface {group_name} {year}-{month_str}"
            })
        
        return requests
    
    def prepare_pressure_requests(self, year, month):
        """Prepara requisições para dados de níveis de pressão"""
        year_dir = str(year)
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)
        
        month_str = f'{month:02d}'
        days = get_days(year, month)
        hours = ['00:00', '06:00', '12:00', '18:00']
        
        # Variáveis padrão para todos os níveis
        standard_vars = [
            'geopotential',
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind',
            'relative_humidity',
        ]
        
        # Definição de grupos de níveis
        pressure_groups = {
            'stratosphere': {
                'levels': ['10', '30', '50'],
                'variables': standard_vars + ['ozone_mass_mixing_ratio']
            },
            'upper_troposphere': {
                'levels': ['100', '150', '200'],
                'variables': standard_vars
            },
            'mid_troposphere': {
                'levels': ['300', '400', '500'],
                'variables': standard_vars
            },
            'lower_troposphere': {
                'levels': ['600', '700', '750'],
                'variables': standard_vars
            },
            'boundary_layer': {
                'levels': ['850', '925', '1000'],
                'variables': standard_vars
            },
        }
        
        requests = []
        for group_name, config in pressure_groups.items():
            output_file = os.path.join(
                year_dir,
                f'era5_pressure_{group_name}_{year}_{month_str}.nc'
            )
            
            request = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': config['variables'],
                'pressure_level': config['levels'],
                'year': str(year),
                'month': month_str,
                'day': days,
                'time': hours,
                'grid': ['0.25', '0.25'],
            }
            
            requests.append({
                'type': 'pressure',
                'output_file': output_file,
                'request': request,
                'description': f"Pressure {group_name} {year}-{month_str}"
            })
        
        return requests
    
    def download_missing_files_for_year(self, year):
        """Baixa apenas os arquivos faltantes de um ano"""
        year_dir = str(year)
        
        # Verifica arquivos faltantes
        missing_items = check_missing_files(year_dir)
        
        if not missing_items:
            self.logger.info(f"Ano {year}: todos os arquivos já existem!")
            return {'completed': 0, 'failed': 0, 'skipped': 0}
        
        self.logger.info(f"Ano {year}: encontrados {len(missing_items)} arquivos faltantes")
        
        # Agrupa por mês para processar
        all_requests = []
        
        # Prepara apenas as requisições necessárias
        for item in missing_items:
            month = item['month']
            if item['type'] == 'surface':
                surface_requests = self.prepare_surface_requests(year, month)
                for req in surface_requests:
                    expected_file = f'era5_surface_{item["group"]}_{year}_{month:02d}.nc'
                    if expected_file in req['output_file']:
                        all_requests.append(req)
                        self.logger.debug(f"Adicionado para download: {req['description']}")
                        break
            elif item['type'] == 'pressure':
                pressure_requests = self.prepare_pressure_requests(year, month)
                for req in pressure_requests:
                    expected_file = f'era5_pressure_{item["group"]}_{year}_{month:02d}.nc'
                    if expected_file in req['output_file']:
                        all_requests.append(req)
                        self.logger.debug(f"Adicionado para download: {req['description']}")
                        break
        
        self.logger.info(f"Total de requisições preparadas: {len(all_requests)}")
        return self.execute_downloads(all_requests)
    
    def execute_downloads(self, all_requests):
        """Executa uma lista de downloads em paralelo"""
        total_requests = len(all_requests)
        self.logger.info(f"Total de requisições preparadas: {total_requests}")
        
        completed = 0
        failed = 0
        skipped = 0
        
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_request = {
                executor.submit(self.download_single_request, req): req 
                for req in all_requests
            }
            
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        completed += 1
                    elif result['status'] == 'error':
                        failed += 1
                    else:
                        skipped += 1
                    
                    total_done = completed + failed + skipped
                    self.logger.info(
                        f"Progresso: {total_done}/{total_requests} "
                        f"(Completos: {completed} | Falhas: {failed} | Pulados: {skipped})"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Erro no futuro: {str(e)}")
                    failed += 1
        
        return {'completed': completed, 'failed': failed, 'skipped': skipped}

def main():
    """Função principal"""
    
    # Configurações
    MAX_PARALLEL_DOWNLOADS = 5  
    MAX_REQUESTS_PER_MINUTE = 20
    
    START_YEAR = 1950
    END_YEAR = 2025
    
    # Inicializar downloader
    downloader = ERA5Downloader6Hourly(
        max_parallel_requests=MAX_PARALLEL_DOWNLOADS,
        max_requests_per_minute=MAX_REQUESTS_PER_MINUTE
    )
    
    downloader.logger.info("=" * 60)
    downloader.logger.info("INICIANDO DOWNLOAD ERA5 - DADOS 6-HORÁRIOS")
    downloader.logger.info(f"Período: {START_YEAR}-{END_YEAR}")
    downloader.logger.info("Resolução temporal: 6 horas (00, 06, 12, 18 UTC)")
    downloader.logger.info("Resolução espacial: 1.0 grau")
    downloader.logger.info("Formato: NetCDF")
    downloader.logger.info(f"Downloads paralelos: {MAX_PARALLEL_DOWNLOADS}")
    downloader.logger.info("=" * 60)
    
    # Estatísticas totais
    total_stats = {'completed': 0, 'failed': 0, 'skipped': 0}
    
    # Loop principal
    for year in range(START_YEAR, END_YEAR + 1):
        downloader.logger.info(f"\n{'=' * 40}")
        downloader.logger.info(f"Processando ano: {year}")
        downloader.logger.info(f"{'=' * 40}")
        
        # Validação inicial
        year_dir = str(year)
        if os.path.exists(year_dir):
            corrupted = validate_and_clean_directory(year_dir, downloader.logger)
            if corrupted:
                downloader.logger.warning(f"Removidos {len(corrupted)} arquivos corrompidos de {year}")
        
        try:
            stats = downloader.download_missing_files_for_year(year)
            
            total_stats['completed'] += stats['completed']
            total_stats['failed'] += stats['failed']
            total_stats['skipped'] += stats['skipped']
            
            downloader.logger.info(f"Ano {year} completo: {stats}")
            
        except Exception as e:
            downloader.logger.error(f"Erro ao processar ano {year}: {str(e)}")
        
        time.sleep(2)
    
    # Relatório final
    downloader.logger.info("\n" + "=" * 60)
    downloader.logger.info("DOWNLOAD COMPLETO!")
    downloader.logger.info(f"Total de arquivos baixados: {total_stats['completed']}")
    downloader.logger.info(f"Total de arquivos já existentes: {total_stats['skipped']}")
    downloader.logger.info(f"Total de falhas: {total_stats['failed']}")
    downloader.logger.info("=" * 60)

if __name__ == "__main__":
    main()
