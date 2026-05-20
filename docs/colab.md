# Running InterruptBench on Google Colab

## Runtime
- Runtime → Change runtime type → T4 GPU

## Setup cells

\`\`\`python
# 1. Verify GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
\`\`\`

\`\`\`bash
# 2. Clone
!git clone https://github.com/YOUR_USERNAME/interruptbench.git
%cd interruptbench
\`\`\`

\`\`\`bash
# 3. Install (pinned versions)
!pip install -q trl==0.17.0 transformers==4.51.3 accelerate==1.6.0 datasets peft bitsandbytes
\`\`\`

\`\`\`bash
# 4. Dry-run
!python scripts/train_grpo.py --dry-run --episodes 10
\`\`\`

## Expected output
- `Dry run succeeded.`
- Reward sample ~0.5 for untrained model

## Known issues
- TRL/Transformers mismatch: pin trl==0.17.0 + transformers==4.51.3
- Ignore numpy/numba/cudf dependency warnings — harmless
