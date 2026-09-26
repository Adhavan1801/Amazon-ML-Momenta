"""Launch the pipeline as a SageMaker Processing job (pay only while it runs).

    pip install boto3
    python sagemaker/launch_processing_job.py --bucket <bucket> --run-name v3 \
        --role arn:aws:iam::<account-id>:role/<SageMakerExecutionRole> [--instance ml.r5.4xlarge]

Expects in S3:  s3://<bucket>/code/  (this folder),  s3://<bucket>/dataset/{train,test}/*.tsv,
                s3://<bucket>/utils/validate_submission.py
Writes:         s3://<bucket>/runs/<run-name>/  (output/*.tsv, run.log, summary.txt, decision.json)
"""
import argparse
import time

import boto3

# AWS Deep Learning Container (CPU, Python 3.11). If this tag is not available in your region,
# pick any "pytorch-training ... cpu-py311" image from the AWS DLC list and pass it with --image.
DEFAULT_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--role", required=True, help="SageMaker execution role ARN (needs S3 read/write)")
    ap.add_argument("--instance", default="ml.r5.4xlarge")
    ap.add_argument("--volume-gb", type=int, default=100)
    ap.add_argument("--region", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-hours", type=float, default=6)
    ap.add_argument("--wait", action="store_true", help="poll until the job finishes")
    a = ap.parse_args()

    sess = boto3.session.Session(region_name=a.region)
    region = sess.region_name
    sm = sess.client("sagemaker")
    image = a.image or DEFAULT_IMAGE.format(region=region)
    b = f"s3://{a.bucket}"
    name = f"ber-{a.run_name}-{time.strftime('%Y%m%d-%H%M%S')}".replace("_", "-")[:63]

    def inp(n, s3, local):
        return {"InputName": n, "S3Input": {"S3Uri": s3, "LocalPath": local, "S3DataType": "S3Prefix",
                                             "S3InputMode": "File", "S3DataDistributionType": "FullyReplicated"}}

    sm.create_processing_job(
        ProcessingJobName=name,
        RoleArn=a.role,
        AppSpecification={"ImageUri": image,
                          "ContainerEntrypoint": ["bash", "/opt/ml/processing/code/sagemaker/run_on_sagemaker.sh"],
                          "ContainerArguments": ["processing", a.run_name]},
        ProcessingInputs=[inp("code", f"{b}/code/", "/opt/ml/processing/code"),
                          inp("dataset", f"{b}/dataset/", "/opt/ml/processing/dataset"),
                          inp("utils", f"{b}/utils/", "/opt/ml/processing/utils")],
        ProcessingOutputConfig={"Outputs": [{"OutputName": "results", "S3Output": {
            "S3Uri": f"{b}/runs/{a.run_name}/", "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob"}}]},
        ProcessingResources={"ClusterConfig": {"InstanceCount": 1, "InstanceType": a.instance,
                                               "VolumeSizeInGB": a.volume_gb}},
        StoppingCondition={"MaxRuntimeInSeconds": int(a.max_hours * 3600)},
    )
    print(f"Started processing job: {name}")
    print(f"Console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/processing-jobs/{name}")
    print(f"Results will be in {b}/runs/{a.run_name}/")
    if a.wait:
        while True:
            d = sm.describe_processing_job(ProcessingJobName=name)
            st = d["ProcessingJobStatus"]
            print(time.strftime("%H:%M:%S"), st, flush=True)
            if st in ("Completed", "Failed", "Stopped"):
                if st != "Completed":
                    print(d.get("FailureReason", ""))
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
