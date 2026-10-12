# CLAUDE.md — monan_ai_preproc (MONAN-AI v3 dataset)

Este repositório converte ERA5 1° (NetCDF, grade regular lat-lon) no dataset
**MONAN-AI v3**: HEALPix `nside=64` (49152 px, ordem **RING**), **235 variáveis**,
6h (00/06/12/18Z), normalizado em memmaps float16 `train/val/test.npy`.

A tarefa típica numa máquina nova é **reconstruir o dataset a partir do ERA5 local
mantendo as estatísticas de normalização congeladas** que vieram do cluster
(ncempa). Os modelos já treinados (núcleos n8/n16/n32/n64, cascata SWIN, DA)
dependem dessas estatísticas: qualquer mudança torna o dataset incompatível.

Guia completo, passo a passo: **`docs/BUILD_V3_DATASET.md`**. Leia antes de rodar.

## Regras invioláveis
1. **NÃO rodar `scripts/02_compute_stats.py`.** Usar sempre
   `reference/stats/healpix_statistics_v3.json` e `reference/stats/healpix_masks_v3.nc`
   (md5 do json: `372443cef4ee4fc97709b5f8e5947a88`; `verify_reference.py stats` checa).
2. **Não editar** `monan_preproc/catalog.py`, `monan_preproc/stats.py`, transforms,
   níveis, filtros de hidrometeoros, nem a ordem das variáveis (alfabética, 235).
3. **Ordem RING** (default do código). Não passar `--nested`. O README antigo dizia
   NESTED — está errado; os arquivos de referência são RING.
4. **Datas dos splits fixas** (iguais a `reference/index/*_index.json`):
   train 1979-01-01..2021-11-30 · val 2022-01-01..2022-11-30 · test 2023-01-01..2024-12-31.
   Se faltar ERA5 para cobrir isso, **pare e reporte** — não encurte datas por conta própria.
5. Se o layout/nomes dos arquivos ERA5 locais forem diferentes do esperado, **adapte
   com uma árvore de symlinks** apontando para os arquivos reais (ver guia, seção 2),
   não alterando o código de conversão. Se as próprias variáveis/dimensões forem
   diferentes (ex.: outra grade, nomes de variáveis distintos), pare e reporte.
6. Cada etapa só avança se a verificação anterior der `ALL PASS`
   (`WARN` é aceitável mas deve ser reportado ao usuário).
7. Não apagar dados existentes; os passos pulam saídas que já existem.

## Caminho rápido
```bash
pip install -r requirements.txt          # ou conda env equivalente (python>=3.10)
export RAW_ROOT=/caminho/era5 OUT_ROOT=/caminho/era5_healpix_v3 \
       NORM_DIR=/caminho/monan_ia_v3/normalized CLIM_DIR=/caminho/monan_ia_v3/climatology \
       JOBS=8 NORM_WORKERS=8
bash scripts/run_local_pipeline.sh check      # deps + md5 das stats
bash scripts/run_local_pipeline.sh cache
bash scripts/run_local_pipeline.sh smoke      # converte 2022-01 e 1979-01 e compara com referência
bash scripts/run_local_pipeline.sh convert    # 1978-12 .. 2024-12 (paralelo, retomável)
bash scripts/run_local_pipeline.sh verify
bash scripts/run_local_pipeline.sh normalize  # usa as stats congeladas
bash scripts/run_local_pipeline.sh clim
bash scripts/run_local_pipeline.sh final      # compara splits com a referência + 06_consistency_checks
```
Rode etapas longas (`convert`, `normalize`, `clim`) com `nohup`/`tmux` e logs em `logs/`.

## O que reportar ao usuário ao final
- saída do `final` (PASS/FAIL), tamanhos de `train/val/test.npy`, caminho de tudo;
- qualquer WARN do `smoke` (diferença pequena vs. referência ⇒ versão/fonte do ERA5);
- meses ausentes/convertidos com erro (logs em `logs/convert_*.log`).
