# Reconstruir o dataset MONAN-AI v3 em outra máquina (stats congeladas)

Objetivo: gerar, a partir do ERA5 1° local, exatamente o mesmo dataset v3 usado
no cluster ncempa — mesmas 235 variáveis, mesma grade HEALPix nside=64 RING,
mesmos splits e **mesmas estatísticas de normalização** — para que os modelos já
treinados possam ser usados/continuados nesta máquina.

## 0. O que vem pronto no git (`reference/`)

| Arquivo | Para quê |
|---|---|
| `reference/stats/healpix_statistics_v3.json` | média/desvio + transform (signed-power/asinh) por variável. **Congelado.** |
| `reference/stats/healpix_masks_v3.nc` | `coverage_mask` / `variation_mask` por variável. **Congelado.** |
| `reference/stats/compute_stats.log` | log original do cálculo (proveniência). |
| `reference/index/{train,val,test}_index.json` | metadados dos splits do cluster (datas, n_timesteps, var_list, ...). |
| `reference/fingerprints/monthly_fingerprint.json` | média/desvio/fração-NaN de t0 de cada variável nos meses 1979-01 e 2022-01 já convertidos. |
| `reference/fingerprints/normalized_fingerprint.json` | média/desvio por variável do 1º timestep de `val.npy` e `test.npy`. |

A climatologia (2 × 67 GB) **não** vai no git: é recalculada no passo 6 a partir do
`train.npy` (determinística: mesmo train ⇒ mesma climatologia).

Tudo isso foi validado no próprio cluster com `scripts/verify_reference.py`
(ALL PASS; diferença máx. 5e-7 nos normalizados).

## 1. Requisitos

- Python ≥ 3.10; `pip install -r requirements.txt`
  (no cluster: numpy 2.2.6, xarray 2025.6.1, healpy 1.19.0, scikit-learn 1.7.2,
  netCDF4 1.7.4, pandas 2.3.3).
- Disco (aprox.):
  - mensais HEALPix (passo 3): ~2.3 GB/mês × 553 meses ≈ **1.3 TB**
  - normalizados: train 1.45 TB + val 31 GB + test 68 GB ≈ **1.55 TB** (de preferência NVMe; o treino lê aleatoriamente)
  - climatologia: 2 × 67 GB ≈ **135 GB**
- RAM: conversão ~4 GB por mês em paralelo (`JOBS`); climatologia pico ~34 GB + buffer (recomendo ≥ 64 GB).
- Tempo de referência: conversão ~5 min/mês/núcleo ⇒ 553 meses com `JOBS=8` ≈ 6 h.

## 2. Layout esperado do ERA5 bruto (`RAW_ROOT`)

Grade regular 1° (181 × 360 típico do CDS; outra ordenação de lat/lon funciona —
o índice de regrid é calculado a partir das coordenadas do arquivo). Dimensões
aceitas: tempo `valid_time` ou `time`; `latitude`/`longitude` ou `lat`/`lon`;
nível `pressure_level` ou `level`. Variáveis com os short names do CDS.

```
RAW_ROOT/
├── {YYYY}/
│   ├── era5_pressure_boundary_layer_{YYYY}_{MM}.nc        r,t,u,v,z @ 850,925,1000
│   ├── era5_pressure_lower_troposphere_{YYYY}_{MM}.nc     r,t,u,v,z @ 600,700,750
│   ├── era5_pressure_mid_troposphere_{YYYY}_{MM}.nc       r,t,u,v,z @ 300,400,500
│   ├── era5_pressure_upper_troposphere_{YYYY}_{MM}.nc     r,t,u,v,z @ 100,150,200
│   ├── era5_pressure_stratosphere_{YYYY}_{MM}.nc          r,t,u,v,z,o3 @ 10,30,50
│   ├── era5_surface_boundary_layer_{YYYY}_{MM}.nc         blh,cape,cin,kx,totalx
│   ├── era5_surface_clouds_{YYYY}_{MM}.nc                 cbh,hcc,lcc,mcc,tcc
│   ├── era5_surface_ocean_ice_{YYYY}_{MM}.nc              istl1-4,siconc
│   ├── era5_surface_pressure_{YYYY}_{MM}.nc               msl
│   ├── era5_surface_snow_{YYYY}_{MM}.nc                   sd
│   ├── era5_surface_snow_processes_{YYYY}_{MM}.nc         rsn
│   ├── era5_surface_soil_moisture_{YYYY}_{MM}.nc          swvl1-4
│   ├── era5_surface_soil_temperature_{YYYY}_{MM}.nc       stl1-4
│   ├── era5_surface_sst_{YYYY}_{MM}.nc                    sst
│   ├── era5_surface_temperature_{YYYY}_{MM}.nc            d2m,t2m
│   ├── era5_surface_water_column_{YYYY}_{MM}.nc           tciw,tclw,tco3,tcrw,tcsw,tcw,tcwv
│   ├── era5_surface_wind_{YYYY}_{MM}.nc                   u10,u100,v10,v100
│   ├── era5_surface_hourly_precipitation_{YYYY}_{MM}.nc   tp            (HORÁRIO, acumulado)
│   ├── era5_surface_hourly_evaporation_{YYYY}_{MM}.nc     e,pev         (horário)
│   ├── era5_surface_hourly_runoff_{YYYY}_{MM}.nc          ro,sro,ssro   (horário)
│   ├── era5_surface_hourly_snow_{YYYY}_{MM}.nc            sf,smlt       (horário)
│   ├── era5_surface_hourly_heat_flux_{YYYY}_{MM}.nc       sshf,slhf     (horário)
│   ├── era5_surface_hourly_radiation_solar_{YYYY}_{MM}.nc ssr,ssrd,ssrdc,tisr,tsr (horário)
│   └── era5_surface_hourly_radiation_thermal_{YYYY}_{MM}.nc str,strd,strdc,ttr    (horário)
├── data_additional_levels/{YYYY}/era5_pressure_additional_levels_{YYYY}_{MM}.nc   r,t,u,v,z @ 250,950
├── data_hydrometeors/{YYYY}/era5_pressure_hydro_precip_{YYYY}_{MM}.nc            crwc,cswc (17 níveis)
├── data_hydrometeors/{YYYY}/era5_pressure_hydro_nonprecip_{YYYY}_{MM}.nc         clwc,ciwc (17 níveis)
├── data_hydrometeors/{YYYY}/era5_pressure_cloud_fraction_{YYYY}_{MM}.nc          cc (17 níveis)
└── data_ozone_extended/{YYYY}/era5_pressure_ozone_extended_{YYYY}_{MM}.nc        o3 @ 100..1000 (14 níveis)
```
(Arquivos extras como `era5_surface_temperature_extremes`, `precipitation_type`,
`vegetation` são ignorados de propósito.)

Detalhes que importam:
- Instantâneos precisam conter 00/06/12/18Z (horário ou 6-horário servem).
- Acumulados são **horários**; cada passo 6h soma `[t-5h, t]`. Para o 1º dia do mês
  o código lê as últimas horas do **mês anterior** — por isso convertemos a partir
  de 1978-12 e o dezembro de 1978 bruto precisa existir (o mesmo vale para cada
  janeiro: o dezembro anterior tem que estar presente).
- Arquivo ausente gera só `WARNING` no log e a variável fica faltando/NaN — o
  passo `verify` pega isso (contagem ≠ 235 ou variável toda NaN).

**Se o layout local for diferente** (outros diretórios ou nomes), crie uma árvore
de symlinks com o layout acima e aponte `RAW_ROOT` para ela. Exemplo:
```bash
mkdir -p /data/era5_v3_links/2020
ln -s /onde/esta/o/arquivo/pressao_bl_2020_01.nc \
      /data/era5_v3_links/2020/era5_pressure_boundary_layer_2020_01.nc
```
Faça um script para gerar os links e confira com `ls` antes de converter.
Se as **variáveis** dentro dos arquivos estiverem distribuídas de outro jeito
(ex.: um arquivo por variável), não altere `catalog.py`: pare e pergunte ao usuário.

## 3. Passo a passo

Defina as variáveis de ambiente uma vez:
```bash
export RAW_ROOT=/caminho/era5
export OUT_ROOT=/caminho/era5_healpix_v3
export NORM_DIR=/caminho/monan_ia_v3/normalized
export CLIM_DIR=/caminho/monan_ia_v3/climatology
export JOBS=8            # meses em paralelo na conversão
export NORM_WORKERS=8    # processos do 03_normalize
```

| # | Comando | Saída / critério |
|---|---|---|
| 1 | `bash scripts/run_local_pipeline.sh check` | deps OK + `ALL PASS` das stats |
| 2 | `bash scripts/run_local_pipeline.sh cache` | `$OUT_ROOT/_cache/regrid_ring_*` e `degrade_ring_*` |
| 3 | `bash scripts/run_local_pipeline.sh smoke` | converte 2022-01 e 1979-01; **ALL PASS** contra `monthly_fingerprint.json` |
| 4 | `nohup bash scripts/run_local_pipeline.sh convert > logs/convert_all.log 2>&1 &` | `$OUT_ROOT/{YYYY}/{YYYY}_{MM}.nc`, 1978-12..2024-12; retomável (pula existentes) |
| 5 | `bash scripts/run_local_pipeline.sh verify` | nenhum mês faltando, todos com 235 vars |
| 6 | `nohup bash scripts/run_local_pipeline.sh normalize > logs/normalize_all.log 2>&1 &` | `$NORM_DIR/{train,val,test}.npy` + `_times.npy` + `_index.json` (+ cópia de `stats.json` e `masks.nc`) |
| 7 | `nohup bash scripts/run_local_pipeline.sh clim > logs/clim_all.log 2>&1 &` | `$CLIM_DIR/climatology_doy_hour_{mean,std}_nside64_v3.nc` |
| 8 | `bash scripts/run_local_pipeline.sh final` | `verify_reference.py normalized` **ALL PASS** + `06_consistency_checks` PASS |

Interpretação do `smoke` (passo 3):
- `PASS` (diferença ≤ 1e-4 do desvio) — ERA5 idêntico ao do cluster.
- `WARN` (1e-4 a 1%) — mesma informação, pequena diferença de fonte/versão (ex.:
  CDS vs ARCO, compressão). Aceitável; reporte ao usuário e siga.
- `FAIL` (> 1%, nomes/contagem/ordem errados) — **pare**. Causas comuns: arquivo
  bruto errado no link, unidade diferente (ex.: acumulado já em mm ou já 6-horário),
  grade não 1°, ordering errado.

Um `normalize` interrompido deve ser refeito para aquele split: apague apenas
`$NORM_DIR/<split>.npy`, `<split>_times.npy` e `<split>_index.json` desse split e rode de novo
(o passo pula splits cujo `_index.json` existe).

## 4. Depois do dataset (opcional, só se o usuário pedir)
Datasets derivados usados por modelos específicos — **não gerar sem pedido**:
- n16/n8 degradados: `scripts/07_degrade_n64_to_n16.py` / `08_degrade_normalized_n64_to_n16.py`
- par base/resíduo da cascata: `scripts/09_upsample_n4_to_n8_residual.py` + `10_compute_residual_scaling.py`

Atenção: os modelos existentes foram treinados com degradação feita **no espaço
normalizado** (script 08); isso tem viés de Jensen conhecido em `tp`, mas para
reproduzir o que os modelos viram é preciso usar o mesmo operador.

## 5. Formato final (contrato com o código de treino)
- `{split}.npy`: `(n_steps, 49152, 235)` float16, z-score pós-transform, clip ±10, NaN→0.
- `var_list`: ordem alfabética (igual a `reference/index/train_index.json`).
- Denormalização: `y = z*std + mean` → inverso do transform → `NaN` onde `coverage_mask==0`.
- Climatologia: `(366 doy, 4 horas, 49152)` por variável, no espaço normalizado.
