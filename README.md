# NSGA-III-Based Multi-Modal Middleware for Cloud-Native Pipeline Orchestration
[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview
This repository contains the implementation developed for my Master's Thesis:

# NSGA-III Based Multi-Modal Middleware for Dynamic Orchestration of Distributed Data Processing Pipelines

The project proposes a middleware system that applies **multi-objective optimization** to orchestrate Directed Acyclic Graph (DAG)-based data processing pipelines. The middleware uses the **NSGA-III evolutionary algorithm** to balance two competing objectives:

- Execution Cost
- End-to-End Latency

The solution is designed for heterogeneous cloud-native environments based on **Kubernetes** and **KubeVirt**, where orchestration decisions involve both virtual machine selection and resource allocation.

## Research Context

Modern data-processing systems frequently combine:

- Batch Processing
- Stream Processing
- Serving Workloads
These multi-modal pipelines introduce complex orchestration challenges because deployment decisions affect both performance and operational cost.

This project investigates the following research question:

#### How can a middleware system, using a Pareto-based evolutionary algorithm (NSGA-III), dynamically orchestrate multi-modal data processing pipelines on Kubernetes/KubeVirt to optimize the trade-off between execution cost and end-to-end latency?

### System Architecture

The middleware follows a three-layer architecture:

### Layer 1 – Logical Pipeline Manager (LPM)

Responsibilities:

- Parse pipeline definitions from YAML
- Construct DAG representation using NetworkX
- Validate task dependencies
- Apply workload scaling factors

### Layer 2 – Optimization Engine

Responsibilities:

- Formulate orchestration as a bi-objective optimization problem
- Execute NSGA-III using the pymoo framework
- Generate Pareto-optimal orchestration plans
- Evaluate cost–latency trade-offs

### Layer 3 – KubeVirt Adapter

Responsibilities:

- Translate deployment plans into Kubernetes/KubeVirt manifests
- Preserve DAG execution dependencies
- Support dry-run and deployment modes




