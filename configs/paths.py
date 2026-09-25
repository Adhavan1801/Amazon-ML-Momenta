"""
Central path definitions for the project.
All paths are relative to the project root (E:\hackathon).
"""
import os

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Original dataset paths
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset", "student_resource", "dataset")

TRAIN_SOURCE1 = os.path.join(DATASET_DIR, "train", "train_source1.tsv")
TRAIN_SOURCE2 = os.path.join(DATASET_DIR, "train", "train_source2.tsv")
TRAIN_SOURCE3 = os.path.join(DATASET_DIR, "train", "train_source3.tsv")
TRAIN_GROUND_TRUTH = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")

TEST_SOURCE1 = os.path.join(DATASET_DIR, "test", "test_source1.tsv")
TEST_SOURCE2 = os.path.join(DATASET_DIR, "test", "test_source2.tsv")
TEST_SOURCE3 = os.path.join(DATASET_DIR, "test", "test_source3.tsv")

# Processed data paths
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

TRAIN_S1_CLEAN = os.path.join(PROCESSED_DIR, "train_source1_clean.parquet")
TRAIN_S2_CLEAN = os.path.join(PROCESSED_DIR, "train_source2_clean.parquet")
TRAIN_S3_CLEAN = os.path.join(PROCESSED_DIR, "train_source3_clean.parquet")

TEST_S1_CLEAN = os.path.join(PROCESSED_DIR, "test_source1_clean.parquet")
TEST_S2_CLEAN = os.path.join(PROCESSED_DIR, "test_source2_clean.parquet")
TEST_S3_CLEAN = os.path.join(PROCESSED_DIR, "test_source3_clean.parquet")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
