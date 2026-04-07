"""
Main Orchestration Script - Royal Enfield App Review Analysis Pipeline
Complete end-to-end analysis: Scraping → Classification → Analysis → Reporting

Usage:
    python main.py                    # Run complete pipeline
    python main.py --skip-scrape      # Use existing data, skip scraping
    python main.py --skip-classify    # Skip classification, use existing classifications
"""

import os
import sys
import json
import pandas as pd
import logging
import argparse
from datetime import datetime
from typing import Optional, Tuple

# Import custom modules
from config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    DEBUG,
    LOGGING_CONFIG,
    validate_config,
)
from scraper import GooglePlayScraper, DataValidator
from classifier import classify_reviews
from analyzer import run_analysis

# ============================================================================
# Setup Logging
# ============================================================================

def setup_logging():
    """Configure logging for the pipeline"""
    
    log_file = os.path.join(PROJECT_ROOT, "logs", f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Use UTF-8 for file handler; stream handler uses replace to avoid emoji crashes on Windows cp1252
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(LOGGING_CONFIG["format"]))
    stream_handler.stream.reconfigure(encoding="utf-8", errors="replace") if hasattr(stream_handler.stream, "reconfigure") else None

    logging.basicConfig(
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            stream_handler,
        ],
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()

# ============================================================================
# Pipeline Functions
# ============================================================================

def step_1_scrape_reviews(force_scrape: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Step 1: Scrape Google Play Store reviews
    
    Args:
        force_scrape: If False, use existing data if available
        
    Returns:
        Tuple of (DataFrame, filepath)
    """
    
    existing_file = f"{RAW_DATA_DIR}/reviews_raw.csv"
    
    # Check if data already exists
    if os.path.exists(existing_file) and not force_scrape:
        logger.info(f"Found existing raw data at {existing_file}, loading...")
        df = pd.read_csv(existing_file)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        logger.info(f"✅ Loaded {len(df)} reviews from existing file")
        return df, existing_file
    
    # Scrape new data
    logger.info("="*70)
    logger.info("STEP 1: SCRAPING GOOGLE PLAY STORE REVIEWS")
    logger.info("="*70)
    
    scraper = GooglePlayScraper()
    df = scraper.scrape_reviews()
    
    # Validate
    validator = DataValidator()
    is_valid, issues = validator.validate(df)
    
    if not is_valid:
        logger.warning(f"Data validation found {len(issues)} issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Save raw data
    filepath = scraper.save_raw_data(df)
    
    # Print summary
    _print_step_summary("Scraping", {
        "total_reviews": len(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}" if 'date' in df.columns else "N/A",
        "ratings": df['rating'].value_counts().to_dict() if 'rating' in df.columns else {},
    })
    
    return df, filepath


def step_2_classify_reviews(df: pd.DataFrame, force_classify: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Step 2: Classify reviews for L1 & L2 intent
    
    Args:
        df: Raw reviews DataFrame
        force_classify: If False, use existing classifications if available
        
    Returns:
        Tuple of (Classified DataFrame, filepath)
    """
    
    existing_file = f"{PROCESSED_DATA_DIR}/reviews_classified.csv"
    
    # Check if classifications already exist
    if os.path.exists(existing_file) and not force_classify:
        logger.info(f"Found existing classifications at {existing_file}, loading...")
        df_classified = pd.read_csv(existing_file)
        logger.info(f"✅ Loaded classifications for {len(df_classified)} reviews")
        return df_classified, existing_file
    
    # Classify reviews
    logger.info("="*70)
    logger.info("STEP 2: CLASSIFYING REVIEWS (L1 & L2 INTENT)")
    logger.info("="*70)
    
    df_classified = classify_reviews(df)
    
    # Save classified data
    output_file = f"{PROCESSED_DATA_DIR}/reviews_classified.csv"
    df_classified.to_csv(output_file, index=False)
    logger.info(f"✅ Classified data saved to {output_file}")
    
    # Print summary
    _print_step_summary("Classification", {
        "total_classified": len(df_classified),
        "l1_distribution": df_classified['l1_intent'].value_counts().to_dict() if 'l1_intent' in df_classified.columns else {},
        "sentiment_distribution": df_classified['sentiment'].value_counts().to_dict() if 'sentiment' in df_classified.columns else {},
    })
    
    return df_classified, output_file


def step_3_run_analysis(df: pd.DataFrame) -> Tuple[dict, str]:
    """
    Step 3: Run comprehensive analysis
    
    Args:
        df: Classified reviews DataFrame
        
    Returns:
        Tuple of (Analysis results dict, filepath)
    """
    
    logger.info("="*70)
    logger.info("STEP 3: RUNNING COMPREHENSIVE ANALYSIS")
    logger.info("="*70)
    
    analysis_results = run_analysis(df)
    
    # Save analysis results
    output_file = f"{OUTPUT_DIR}/analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    logger.info(f"✅ Analysis results saved to {output_file}")
    
    return analysis_results, output_file


def step_4_generate_report(df: pd.DataFrame, analysis_results: dict) -> str:
    """
    Step 4: Generate executive report
    
    Args:
        df: Classified reviews DataFrame
        analysis_results: Analysis results dictionary
        
    Returns:
        Filepath of generated report
    """
    
    logger.info("="*70)
    logger.info("STEP 4: GENERATING EXECUTIVE REPORT")
    logger.info("="*70)
    
    report = _generate_markdown_report(df, analysis_results)
    
    output_file = f"{OUTPUT_DIR}/ANALYSIS_REPORT.md"
    with open(output_file, 'w') as f:
        f.write(report)
    logger.info(f"✅ Report saved to {output_file}")
    
    return output_file


def step_5_export_data(df: pd.DataFrame, analysis_results: dict):
    """
    Step 5: Export data in multiple formats
    
    Args:
        df: Classified reviews DataFrame
        analysis_results: Analysis results dictionary
    """
    
    logger.info("="*70)
    logger.info("STEP 5: EXPORTING DATA")
    logger.info("="*70)
    
    # CSV exports
    csv_files = [
        (f"{OUTPUT_DIR}/reviews_with_intent.csv", df),
        (f"{OUTPUT_DIR}/key_metrics.csv", _metrics_to_dataframe(analysis_results)),
    ]
    
    for filepath, data in csv_files:
        data.to_csv(filepath, index=False)
        logger.info(f"✅ Exported {filepath}")
    
    # JSON export
    json_file = f"{OUTPUT_DIR}/analysis_complete.json"
    with open(json_file, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_reviews": len(df),
                "analysis_completed": True,
            },
            "analysis": analysis_results,
        }, f, indent=2, default=str)
    logger.info(f"✅ Exported {json_file}")


# ============================================================================
# Helper Functions
# ============================================================================

def _print_step_summary(step_name: str, summary: dict):
    """Print a summary of completed step"""
    
    print("\n" + "="*70)
    print(f"✅ {step_name.upper()} SUMMARY")
    print("="*70)
    
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("="*70 + "\n")


def _generate_markdown_report(df: pd.DataFrame, analysis_results: dict) -> str:
    """Generate markdown report from analysis results"""
    
    report = []
    report.append("# Royal Enfield App Review Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Reviews Analyzed**: {len(df)}")
    
    # Key Metrics
    report.append("\n## Key Metrics")
    metrics = analysis_results.get('insights', {}).get('key_metrics', {})
    for key, value in metrics.items():
        report.append(f"- {key}: {value}")
    
    # Rating Distribution
    if 'eda' in analysis_results and 'rating_analysis' in analysis_results['eda']:
        rating_analysis = analysis_results['eda']['rating_analysis']
        report.append("\n## Rating Distribution")
        
        segments = rating_analysis.get('segments', {})
        for segment, data in segments.items():
            report.append(f"- {segment}: {data['count']} reviews ({data['percent']}%)")
    
    # Top Issues
    report.append("\n## Top Issues (Priority)")
    for issue in analysis_results.get('insights', {}).get('top_issues', []):
        report.append(f"\n### {issue['topic']} (Priority: {issue['impact']})")
        report.append(f"- Sample: {issue['sample'][:100]}...")
    
    # Strengths
    report.append("\n## Strengths")
    for strength in analysis_results.get('insights', {}).get('strengths', []):
        report.append(f"- {strength['topic']}: Avg rating {strength['avg_rating']}/5")
    
    # Recommendations
    report.append("\n## Recommendations")
    for rec in analysis_results.get('insights', {}).get('recommendations', []):
        report.append(f"\n### [{rec['priority']}] {rec['action']}")
        report.append(f"- Rationale: {rec['rationale']}")
        report.append(f"- Timeline: {rec['timeline']}")
        report.append(f"- Impact: {rec['impact']}")
    
    # Limitations
    report.append("\n## Limitations & Caveats")
    report.append("- This analysis is based on public reviews which may skew toward extreme sentiments")
    report.append("- Classifications are AI-generated and may have interpretation errors")
    report.append("- Sentiment correlation with actual user behavior should be validated with telemetry")
    
    return "\n".join(report)


def _metrics_to_dataframe(analysis_results: dict) -> pd.DataFrame:
    """Convert metrics to DataFrame for export"""
    
    metrics = analysis_results.get('insights', {}).get('key_metrics', {})
    return pd.DataFrame([metrics])


# ============================================================================
# Main Pipeline Orchestration
# ============================================================================

def run_full_pipeline(skip_scrape: bool = False, skip_classify: bool = False, force: bool = False):
    """
    Run the complete analysis pipeline
    
    Args:
        skip_scrape: Skip scraping step if data exists
        skip_classify: Skip classification step if classifications exist
        force: Force all steps regardless of existing data
    """
    
    try:
        # Validate configuration
        validate_config()
        logger.info("✅ Configuration validated")
        
        # STEP 1: Scrape Reviews
        logger.info("\n" + "="*70)
        logger.info("PIPELINE START")
        logger.info("="*70)
        
        force_scrape = force or not skip_scrape
        df_raw, raw_file = step_1_scrape_reviews(force_scrape=force_scrape)
        
        # STEP 2: Classify Reviews
        force_classify = force or not skip_classify
        df_classified, classified_file = step_2_classify_reviews(df_raw, force_classify=force_classify)
        
        # STEP 3: Run Analysis
        analysis_results, analysis_file = step_3_run_analysis(df_classified)
        
        # STEP 4: Generate Report
        report_file = step_4_generate_report(df_classified, analysis_results)
        
        # STEP 5: Export Data
        step_5_export_data(df_classified, analysis_results)
        
        # Final Summary
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE - ALL STEPS SUCCESSFUL")
        logger.info("="*70)
        
        logger.info(f"\nOutput Files:")
        logger.info(f"  - Raw Data: {raw_file}")
        logger.info(f"  - Classified Data: {classified_file}")
        logger.info(f"  - Analysis Results: {analysis_file}")
        logger.info(f"  - Report: {report_file}")
        logger.info(f"  - All files: {OUTPUT_DIR}/")
        
        logger.info("\n" + "="*70)
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        return 1


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Royal Enfield App Review Analysis Pipeline"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping, use existing data"
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip classification, use existing classifications"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force all steps regardless of existing data"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    exit_code = run_full_pipeline(
        skip_scrape=args.skip_scrape,
        skip_classify=args.skip_classify,
        force=args.force,
    )
    
    sys.exit(exit_code)
