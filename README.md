# FinReg-RAG 🏦

> A production-grade Retrieval Augmented Generation system for financial regulatory documents — built with hybrid search, cross-encoder reranking, citation enforcement, and automated evaluation.

![CI Pipeline](https://github.com/SatvikAR21/finreg-rag/actions/workflows/eval.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![LangChain](https://img.shields.io/badge/LangChain-0.1-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What This System Does

FinReg-RAG lets compliance officers, analysts, and engineers ask plain-English questions over dense financial regulatory documents and receive **cited, hallucination-free answers** grounded strictly in the source material.

**Example:**
> *"What is the minimum CET1 capital ratio required under Basel III?"*
> → *"Banks must maintain a minimum Common Equity Tier 1 (CET1) ratio of 4.5% of risk-weighted assets. [1]"*
> → Source: Basel III Framework, Page 12

If the documents don't support an answer, the system **refuses to respond** rather than hallucinate.

---

## Evaluation Results

| Metric | Score |
|---|---|
| Overall Evaluation Score | **97.5%** |
| Citation Compliance | **100%** |
| Hallucination Rate | **0%** |
| Refusal Accuracy | **100%** |
| Average End-to-End Latency | **2,512ms** |

*Evaluated against a 10-case RAGAS harness, gated into CI/CD on every push.*

---

## System Architecture
