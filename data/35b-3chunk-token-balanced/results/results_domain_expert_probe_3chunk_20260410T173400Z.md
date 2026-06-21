# Domain Expert Probe 3-Chunk Analysis

- Model: `Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive Q8_0`
- Capture dir: `/workspace/consciousness-experiment/experiments/qwen3.5-35b-a3b-huahua-strangeloop/results/20260410T173400Z_domain_expert_probe_3chunk_balanced_gen_n2048`
- Highlight expert: `114`

- Mean prefill entropy: `0.957522`
- Mean generation-trimmed entropy: `0.952677`

## domain_expert_probe_20A_chunk

- Prompt tokens: `446`
- Generated tokens: `2048`
- Trimmed generated tokens: `2048`
- Prefill entropy mean: `0.957452`
- Generation-trimmed entropy mean: `0.952893`
- Entropy delta (gen_trim - prefill): `-0.004559`
- Top-W Jaccard prefill vs generation-trimmed: `0.000000`
- Top-S Jaccard prefill vs generation-trimmed: `0.066667`
- E114 prefill W/S/Q: `0.005756` / `0.046076` / `0.103020`
- E114 generation-trimmed W/S/Q: `0.005652` / `0.046484` / `0.108754`
- E114 prefill rank by W/S: `30` / `28`
- E114 generation-trimmed rank by W/S: `21` / `13`

Top generation-trimmed experts by W:
- `#1` E169: W `0.008694`, S `0.055078`, Q `0.115115`
- `#2` E252: W `0.008351`, S `0.057300`, Q `0.103436`
- `#3` E94: W `0.007206`, S `0.054468`, Q `0.110505`
- `#4` E248: W `0.007096`, S `0.048901`, Q `0.113745`
- `#5` E82: W `0.006743`, S `0.054370`, Q `0.111180`
- `#6` E116: W `0.006719`, S `0.053625`, Q `0.121204`
- `#7` E146: W `0.006626`, S `0.053955`, Q `0.107201`
- `#8` E8: W `0.006567`, S `0.055005`, Q `0.109080`

Top generation-trimmed experts by S:
- `#1` E252: W `0.008351`, S `0.057300`, Q `0.103436`
- `#2` E169: W `0.008694`, S `0.055078`, Q `0.115115`
- `#3` E8: W `0.006567`, S `0.055005`, Q `0.109080`
- `#4` E94: W `0.007206`, S `0.054468`, Q `0.110505`
- `#5` E82: W `0.006743`, S `0.054370`, Q `0.111180`
- `#6` E146: W `0.006626`, S `0.053955`, Q `0.107201`
- `#7` E116: W `0.006719`, S `0.053625`, Q `0.121204`
- `#8` E28: W `0.006286`, S `0.049609`, Q `0.102164`

## domain_expert_probe_20B_chunk

- Prompt tokens: `446`
- Generated tokens: `2048`
- Trimmed generated tokens: `2048`
- Prefill entropy mean: `0.956576`
- Generation-trimmed entropy mean: `0.950346`
- Entropy delta (gen_trim - prefill): `-0.006230`
- Top-W Jaccard prefill vs generation-trimmed: `0.230769`
- Top-S Jaccard prefill vs generation-trimmed: `0.230769`
- E114 prefill W/S/Q: `0.004381` / `0.034585` / `0.113229`
- E114 generation-trimmed W/S/Q: `0.005765` / `0.046411` / `0.107316`
- E114 prefill rank by W/S: `82` / `82`
- E114 generation-trimmed rank by W/S: `25` / `21`

Top generation-trimmed experts by W:
- `#1` E248: W `0.010151`, S `0.065735`, Q `0.115362`
- `#2` E210: W `0.008600`, S `0.066504`, Q `0.110080`
- `#3` E34: W `0.008512`, S `0.061340`, Q `0.107841`
- `#4` E100: W `0.008495`, S `0.052014`, Q `0.108583`
- `#5` E220: W `0.008444`, S `0.056775`, Q `0.113126`
- `#6` E56: W `0.008337`, S `0.056750`, Q `0.114093`
- `#7` E224: W `0.008107`, S `0.058203`, Q `0.109765`
- `#8` E17: W `0.007550`, S `0.054236`, Q `0.111812`

Top generation-trimmed experts by S:
- `#1` E210: W `0.008600`, S `0.066504`, Q `0.110080`
- `#2` E248: W `0.010151`, S `0.065735`, Q `0.115362`
- `#3` E34: W `0.008512`, S `0.061340`, Q `0.107841`
- `#4` E245: W `0.007402`, S `0.060059`, Q `0.099892`
- `#5` E228: W `0.007055`, S `0.059265`, Q `0.111715`
- `#6` E224: W `0.008107`, S `0.058203`, Q `0.109765`
- `#7` E220: W `0.008444`, S `0.056775`, Q `0.113126`
- `#8` E56: W `0.008337`, S `0.056750`, Q `0.114093`

## domain_expert_probe_20C_chunk

- Prompt tokens: `446`
- Generated tokens: `1838`
- Trimmed generated tokens: `1838`
- Prefill entropy mean: `0.958537`
- Generation-trimmed entropy mean: `0.954792`
- Entropy delta (gen_trim - prefill): `-0.003745`
- Top-W Jaccard prefill vs generation-trimmed: `0.066667`
- Top-S Jaccard prefill vs generation-trimmed: `0.000000`
- E114 prefill W/S/Q: `0.004358` / `0.033913` / `0.107973`
- E114 generation-trimmed W/S/Q: `0.006598` / `0.050571` / `0.106468`
- E114 prefill rank by W/S: `89` / `91`
- E114 generation-trimmed rank by W/S: `7` / `6`

Top generation-trimmed experts by W:
- `#1` E146: W `0.008362`, S `0.060079`, Q `0.117144`
- `#2` E116: W `0.007442`, S `0.057481`, Q `0.117168`
- `#3` E28: W `0.007221`, S `0.054516`, Q `0.106159`
- `#4` E110: W `0.007076`, S `0.046205`, Q `0.122176`
- `#5` E5: W `0.007069`, S `0.057168`, Q `0.112152`
- `#6` E82: W `0.006843`, S `0.050354`, Q `0.114128`
- `#7` E114: W `0.006598`, S `0.050571`, Q `0.106468`
- `#8` E228: W `0.006577`, S `0.048871`, Q `0.116706`

Top generation-trimmed experts by S:
- `#1` E146: W `0.008362`, S `0.060079`, Q `0.117144`
- `#2` E116: W `0.007442`, S `0.057481`, Q `0.117168`
- `#3` E5: W `0.007069`, S `0.057168`, Q `0.112152`
- `#4` E8: W `0.006491`, S `0.055522`, Q `0.109132`
- `#5` E28: W `0.007221`, S `0.054516`, Q `0.106159`
- `#6` E114: W `0.006598`, S `0.050571`, Q `0.106468`
- `#7` E82: W `0.006843`, S `0.050354`, Q `0.114128`
- `#8` E228: W `0.006577`, S `0.048871`, Q `0.116706`

