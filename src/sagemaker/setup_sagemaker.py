"""
Setup SageMaker IAM role and Notebook Instance.

Run this ONCE from your local machine to create:
  1. IAM Role with S3 + SageMaker permissions
  2. SageMaker Notebook Instance (ml.g4dn.xlarge with T4 GPU)

Usage:
    python -m src.sagemaker.setup_sagemaker --create-role
    python -m src.sagemaker.setup_sagemaker --create-notebook
    python -m src.sagemaker.setup_sagemaker --create-all
    python -m src.sagemaker.setup_sagemaker --status
    python -m src.sagemaker.setup_sagemaker --stop-notebook
    python -m src.sagemaker.setup_sagemaker --start-notebook
"""

import argparse
import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, ".")
from configs.sagemaker_config import (
    AWS_REGION,
    S3_BUCKET,
    SAGEMAKER_ROLE_NAME,
    SAGEMAKER_ROLE_ARN,
    NOTEBOOK_INSTANCE_NAME,
    NOTEBOOK_INSTANCE_TYPE_GPU,
    NOTEBOOK_VOLUME_SIZE_GB,
)


# ══════════════════════════════════════════════════════════════════════
# IAM ROLE
# ══════════════════════════════════════════════════════════════════════

# Trust policy: allows SageMaker service to assume this role
TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def create_sagemaker_role():
    """Create an IAM role for SageMaker with S3 access."""
    iam = boto3.client("iam", region_name=AWS_REGION)

    # Check if role already exists
    try:
        existing = iam.get_role(RoleName=SAGEMAKER_ROLE_NAME)
        role_arn = existing["Role"]["Arn"]
        print(f"✓ Role already exists: {role_arn}")
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    print(f"Creating IAM role: {SAGEMAKER_ROLE_NAME}...")

    # Create the role
    response = iam.create_role(
        RoleName=SAGEMAKER_ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
        Description="SageMaker execution role for Momenta entity resolution hackathon",
        MaxSessionDuration=43200,  # 12 hours
    )
    role_arn = response["Role"]["Arn"]
    print(f"  Created role: {role_arn}")

    # Attach managed policies
    policies_to_attach = [
        # Full SageMaker access (notebooks, training, endpoints)
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        # S3 access for reading data and writing models
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        # CloudWatch for logging
        "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
    ]

    for policy_arn in policies_to_attach:
        iam.attach_role_policy(RoleName=SAGEMAKER_ROLE_NAME, PolicyArn=policy_arn)
        policy_name = policy_arn.split("/")[-1]
        print(f"  Attached: {policy_name}")

    print(f"✓ Role ready: {role_arn}")
    return role_arn


# ══════════════════════════════════════════════════════════════════════
# NOTEBOOK INSTANCE
# ══════════════════════════════════════════════════════════════════════

LIFECYCLE_SCRIPT = """#!/bin/bash
set -e

echo "=== Installing ML dependencies ==="

# Install into the pytorch_p310 conda env (has CUDA + PyTorch pre-installed)
source activate pytorch_p310

pip install --quiet --upgrade \\
    sentence-transformers==3.4.1 \\
    transformers==4.47.1 \\
    accelerate==1.2.1 \\
    faiss-gpu==1.7.2 \\
    datasets==3.2.0 \\
    pyarrow==18.1.0 \\
    lightgbm==4.5.0 \\
    scikit-learn==1.6.0 \\
    pandas==2.2.3 \\
    scipy==1.15.0 \\
    rapidfuzz==3.11.0 \\
    jellyfish==1.1.3 \\
    tqdm \\
    boto3

echo "=== Done ==="
"""


def create_lifecycle_config():
    """Create a lifecycle configuration that installs dependencies on notebook start."""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    config_name = f"{NOTEBOOK_INSTANCE_NAME}-lifecycle"

    try:
        sm.describe_notebook_instance_lifecycle_config(
            NotebookInstanceLifecycleConfigName=config_name
        )
        print(f"✓ Lifecycle config already exists: {config_name}")
        return config_name
    except ClientError:
        pass

    import base64
    script_b64 = base64.b64encode(LIFECYCLE_SCRIPT.encode()).decode()

    sm.create_notebook_instance_lifecycle_config(
        NotebookInstanceLifecycleConfigName=config_name,
        OnStart=[{"Content": script_b64}],
    )
    print(f"✓ Created lifecycle config: {config_name}")
    return config_name


def create_notebook_instance(role_arn: str):
    """Create a SageMaker Notebook Instance with GPU."""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)

    # Check if already exists
    try:
        existing = sm.describe_notebook_instance(
            NotebookInstanceName=NOTEBOOK_INSTANCE_NAME
        )
        status = existing["NotebookInstanceStatus"]
        print(f"✓ Notebook already exists: {NOTEBOOK_INSTANCE_NAME} (status: {status})")
        if status == "Stopped":
            print("  → Use --start-notebook to start it")
        return
    except ClientError as e:
        if "RecordNotFound" not in str(e):
            raise

    # Create lifecycle config
    lifecycle_name = create_lifecycle_config()

    print(f"\nCreating notebook instance: {NOTEBOOK_INSTANCE_NAME}")
    print(f"  Instance type: {NOTEBOOK_INSTANCE_TYPE_GPU}")
    print(f"  Volume size:   {NOTEBOOK_VOLUME_SIZE_GB} GB")
    print(f"  Role:          {role_arn}")

    sm.create_notebook_instance(
        NotebookInstanceName=NOTEBOOK_INSTANCE_NAME,
        InstanceType=NOTEBOOK_INSTANCE_TYPE_GPU,
        RoleArn=role_arn,
        VolumeSizeInGB=NOTEBOOK_VOLUME_SIZE_GB,
        LifecycleConfigName=lifecycle_name,
        DirectInternetAccess="Enabled",
        RootAccess="Enabled",
        PlatformIdentifier="notebook-al2-v2",  # Amazon Linux 2
    )
    print("  ⏳ Creating... (takes 3-5 minutes)")

    # Wait for it to become InService
    print("  Waiting for notebook to start...")
    waiter = sm.get_waiter("notebook_instance_in_service")
    waiter.wait(
        NotebookInstanceName=NOTEBOOK_INSTANCE_NAME,
        WaiterConfig={"Delay": 30, "MaxAttempts": 20},
    )

    notebook = sm.describe_notebook_instance(
        NotebookInstanceName=NOTEBOOK_INSTANCE_NAME
    )
    url = notebook.get("Url", "")
    print(f"\n✓ Notebook is ready!")
    print(f"  URL: https://{url}")
    print(f"\n  ⚠️  REMEMBER: Stop the notebook when not in use to save credits!")
    print(f"     python -m src.sagemaker.setup_sagemaker --stop-notebook")


def start_notebook():
    """Start a stopped notebook instance."""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        status = sm.describe_notebook_instance(
            NotebookInstanceName=NOTEBOOK_INSTANCE_NAME
        )["NotebookInstanceStatus"]

        if status == "InService":
            url = sm.describe_notebook_instance(
                NotebookInstanceName=NOTEBOOK_INSTANCE_NAME
            )["Url"]
            print(f"✓ Notebook is already running: https://{url}")
            return

        if status != "Stopped":
            print(f"⚠️  Notebook is in state '{status}', cannot start. Wait and retry.")
            return

        sm.start_notebook_instance(NotebookInstanceName=NOTEBOOK_INSTANCE_NAME)
        print(f"⏳ Starting {NOTEBOOK_INSTANCE_NAME}... (takes 2-3 minutes)")

        waiter = sm.get_waiter("notebook_instance_in_service")
        waiter.wait(
            NotebookInstanceName=NOTEBOOK_INSTANCE_NAME,
            WaiterConfig={"Delay": 30, "MaxAttempts": 15},
        )

        url = sm.describe_notebook_instance(
            NotebookInstanceName=NOTEBOOK_INSTANCE_NAME
        )["Url"]
        print(f"✓ Notebook started: https://{url}")

    except ClientError as e:
        print(f"❌ Error: {e}")


def stop_notebook():
    """Stop a running notebook instance (saves credits!)."""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        status = sm.describe_notebook_instance(
            NotebookInstanceName=NOTEBOOK_INSTANCE_NAME
        )["NotebookInstanceStatus"]

        if status == "Stopped":
            print(f"✓ Notebook is already stopped.")
            return

        if status != "InService":
            print(f"⚠️  Notebook is in state '{status}', cannot stop.")
            return

        sm.stop_notebook_instance(NotebookInstanceName=NOTEBOOK_INSTANCE_NAME)
        print(f"⏳ Stopping {NOTEBOOK_INSTANCE_NAME}...")

        waiter = sm.get_waiter("notebook_instance_stopped")
        waiter.wait(
            NotebookInstanceName=NOTEBOOK_INSTANCE_NAME,
            WaiterConfig={"Delay": 15, "MaxAttempts": 20},
        )
        print(f"✓ Notebook stopped. No more charges until you start it again.")

    except ClientError as e:
        print(f"❌ Error: {e}")


def get_status():
    """Print current status of the notebook instance."""
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        notebook = sm.describe_notebook_instance(
            NotebookInstanceName=NOTEBOOK_INSTANCE_NAME
        )
        status = notebook["NotebookInstanceStatus"]
        instance_type = notebook["InstanceType"]
        url = notebook.get("Url", "N/A")

        print(f"\n📊 Notebook Status")
        print(f"   Name:     {NOTEBOOK_INSTANCE_NAME}")
        print(f"   Status:   {status}")
        print(f"   Type:     {instance_type}")
        if status == "InService":
            print(f"   URL:      https://{url}")
            cost_info = {
                "ml.g4dn.xlarge": 0.736,
                "ml.g5.xlarge": 1.408,
                "ml.m5.xlarge": 0.269,
            }
            cost = cost_info.get(instance_type, 0)
            print(f"   Cost:     ${cost:.3f}/hr")
            print(f"\n   ⚠️  Stop when done: python -m src.sagemaker.setup_sagemaker --stop-notebook")
        elif status == "Stopped":
            print(f"   💤 Not incurring charges.")

    except ClientError:
        print(f"❌ Notebook '{NOTEBOOK_INSTANCE_NAME}' not found.")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Setup SageMaker for entity resolution")
    parser.add_argument("--create-role", action="store_true", help="Create IAM role")
    parser.add_argument("--create-notebook", action="store_true", help="Create notebook instance")
    parser.add_argument("--create-all", action="store_true", help="Create role + notebook")
    parser.add_argument("--start-notebook", action="store_true", help="Start stopped notebook")
    parser.add_argument("--stop-notebook", action="store_true", help="Stop running notebook")
    parser.add_argument("--status", action="store_true", help="Show notebook status")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.create_all or args.create_role:
        role_arn = create_sagemaker_role()

    if args.create_all or args.create_notebook:
        if not (args.create_all or args.create_role):
            role_arn = SAGEMAKER_ROLE_ARN
        create_notebook_instance(role_arn)

    if args.start_notebook:
        start_notebook()

    if args.stop_notebook:
        stop_notebook()

    if args.status:
        get_status()


if __name__ == "__main__":
    main()
