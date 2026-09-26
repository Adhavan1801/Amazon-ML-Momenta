# Running on AWS SageMaker AI + S3

The pipeline is **CPU + RAM bound** (text processing, joins, LightGBM), so pick a **CPU instance with lots of RAM**. GPU instances cost more and won't be faster here.

| Instance | vCPU / RAM | Use |
|---|---|---|
| **ml.r5.4xlarge** (recommended) | 16 / 128 GB | Full run in ~45–60 min |
| ml.m5.4xlarge | 16 / 64 GB | Also fine (~1 h) |
| ml.m5.2xlarge | 8 / 32 GB | Cheaper, ~2 h |

Use the **Mumbai region (ap-south-1)** if it's closest. Check the hourly price on the SageMaker pricing page, and **stop the instance when you're done**. You pay per hour while it runs.

> **Quota check first:** new AWS accounts often have a quota of **0** for bigger SageMaker instance types.
> Go to AWS Console → **Service Quotas** → *Amazon SageMaker* → search the instance name
> (e.g. "ml.r5.4xlarge for JupyterLab spaces" for option A, or "ml.r5.4xlarge for processing job usage" for option B)
> → **Request increase to 1**. It can take a few hours to a day.

---

## Step 1 — Put data and code in S3 (from your Windows PC)

Install the **AWS CLI v2** and run `aws configure` (access key, secret key, region `ap-south-1`).

```bat
REM 1) create a bucket (bucket names are global; pick your own unique name)
aws s3 mb s3://ber-sai-ml-challenge --region ap-south-1

REM 2) upload the dataset (~2.4 GB)
aws s3 sync "D:\ML_Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset" s3://ber-sai-ml-challenge/dataset/ --exclude "*.DS_Store"

REM 3) upload the validator and the code
aws s3 cp "D:\ML_Amazon\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py" s3://ber-sai-ml-challenge/utils/
aws s3 sync "D:\ML_Amazon\code\business_entity_resolution" s3://ber-sai-ml-challenge/code/ --exclude "*__pycache__*"
```

S3 layout afterwards:

```
s3://ber-sai-ml-challenge/
  dataset/train/*.tsv   dataset/test/*.tsv
  utils/validate_submission.py
  code/ (README.md, requirements.txt, src/, sagemaker/)
  runs/<run-name>/output/      <- results land here
```

---

## Option A (simplest) — SageMaker Studio, JupyterLab space

1. AWS Console → **SageMaker AI** → **Domains** → create a domain with **Quick setup** (one-time, a few minutes).
2. Open **Studio** → **JupyterLab** → **Create JupyterLab space**:
   - Instance: **ml.r5.4xlarge**
   - Storage: **100 GB** (the pipeline needs ~40 GB of scratch space)
   - Image: the default *SageMaker Distribution*
   - **Run space** → **Open JupyterLab** → **File → New → Terminal**
3. In the terminal:

```bash
aws s3 sync s3://ber-sai-ml-challenge/code/ ~/ber/code/
bash ~/ber/code/sagemaker/run_on_sagemaker.sh s3://ber-sai-ml-challenge v3
```

The script installs Python 3.11 + requirements, downloads the dataset, runs the whole pipeline, validates, and uploads results to `s3://ber-sai-ml-challenge/runs/v3/`.

4. Download the result to your PC:

```bat
aws s3 cp s3://ber-sai-ml-challenge/runs/v3/output/matching_results.tsv D:\ML_Amazon\output_sm\
```

5. **Stop the space** (Studio → JupyterLab → Stop) so it stops billing.

Tip: the terminal survives closing the browser tab only while the space runs. For long runs use
`nohup bash ~/ber/code/sagemaker/run_on_sagemaker.sh s3://ber-sai-ml-challenge v3 > ~/run.log 2>&1 &`
and check with `tail -f ~/run.log`.

---

## Option B — SageMaker Processing Job (pay only while it runs, no notebook)

From any machine with the SageMaker Python SDK (`pip install sagemaker boto3`), e.g. your PC or a small notebook:

```bash
python sagemaker/launch_processing_job.py --bucket ber-sai-ml-challenge --run-name v3 \
       --role arn:aws:iam::<account-id>:role/<SageMakerExecutionRole> --instance ml.r5.4xlarge
```

It starts a job that downloads `code/`, `dataset/` and `utils/` from S3, runs the same script, and uploads
`runs/v3/output/` (plus logs and the eval score) back to S3. Logs appear in **CloudWatch** (the job page links to them).
The execution role needs read/write access to the bucket; the one Quick setup creates works.

---

## Running only part of the pipeline

Same flags as locally (see README.md), e.g. after changing `model.py`:

```bash
python ~/ber/code/src/run_all.py --data ~/ber/dataset --work ~/ber/work --out ~/ber/output --mem-gb 120 --from train1
```

`~/ber/work` stays on the space's disk between sessions (option A), so you can re-run later steps without redoing the slow ones.

---

## GPU add-on: cross-encoder (4.3) and Qwen2.5-7B judge (4.4) on borderline pairs

The CPU pipeline ends with `export_borderline`. It writes `<work>/borderline/train.parquet` (held-out borderline
pairs **with labels**) and `test.parquet` (test borderline pairs), each with both records' raw name, address and country.
Only these pairs (a few % of all candidates) go to the GPU models:

1. Start a GPU space or instance, e.g. **ml.g5.xlarge** (1× A10G 24 GB). Copy `<work>/borderline/` there
   (e.g. `aws s3 sync ~/ber/work/borderline s3://<bucket>/runs/<run>/borderline/`, then sync it down on the GPU box).
2. Install: `pip install -r sagemaker/gpu/requirements-gpu.txt` (optionally `pip install vllm` for faster Qwen).
3. Cross-encoder (xlm-roberta-base, MIT; ~20–40 min):
   `python sagemaker/gpu/cross_encoder.py --dir <borderline dir>`  → `ce_train.parquet`, `ce_test.parquet`
4. Optional, Qwen2.5-7B-Instruct (Apache-2.0), only on pairs still uncertain after the cross-encoder:
   `python sagemaker/gpu/qwen_judge.py --dir <borderline dir>`  → `llm_train.parquet`, `llm_test.parquet`
5. Copy the `ce_*` / `llm_*` files back into `<work>/borderline/` on the CPU box and run
   `python src/run_all.py ... --only combine`, or directly `python src/cli.py combine` with the same `BER_*` env.
   `combine` tunes the blend weights and threshold on the held-out pairs and prints
   `held-out F0.5 without re-ranker=… with=…`. It rewrites `matching_results.tsv` in `--out`.
   **Submit it only if "with" beats "without".**

Both GPU models only see the two records being compared: no external data or lookups, which is within the rules.
