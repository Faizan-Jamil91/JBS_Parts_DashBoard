import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# ==============================================
#            STREAMLIT DASHBOARD
#   AWAN Parts Intelligence Management System
# ==============================================

# Page configuration
st.set_page_config(
    page_title="AWAN Parts Intelligence System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 4px solid #1e3a8a;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1e3a8a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin: 0.5rem 0;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #4b5563;
        font-weight: 600;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .company-badge {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
#                1. DATA LOADING
# ==============================================

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    sheet_url = (
        "https://docs.google.com/spreadsheets/d/"
        "1Q2n-Y-vaSwGj4vifJX8idpHIquzkspLgRCo_fj0b0Y0/"
        "export?format=csv&gid=442319440"
    )
    
    try:
        df = pd.read_csv(sheet_url, header=1)
        # Remove the first unnamed column
        df = df.drop(df.columns[0], axis=1)
        
        # Data cleaning
        df['Incident Report Date from CRM'] = pd.to_datetime(
            df['Incident Report Date from CRM'], 
            errors='coerce'
        )
        df['Received in Days'] = pd.to_numeric(df['Received in Days'], errors='coerce')
        df = df.dropna(subset=['Region'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_metric_card(label, value, icon="📊"):
    """Helper function to create consistent metric cards"""
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """
    return card_html

# ==============================================
#                DASHBOARD LAYOUT
# ==============================================

def main():
    # Enhanced Header Section
    st.markdown('<div class="company-badge">AWAN INDUSTRIAL SOLUTIONS</div>', 
                unsafe_allow_html=True)
    
    st.markdown(
        '<div class="main-header">🔧 AWAN Parts Intelligence System</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<div class="sub-header">Comprehensive Tag Number Tracking & Inventory Analytics Dashboard</div>', 
        unsafe_allow_html=True
    )
    
    # Load data with progress indicator
    with st.spinner('🔄 Loading AWAN Parts Data...'):
        df = load_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check the data source.")
        return
    
    if df.empty:
        st.warning("⚠️ No data available with the current filters.")
        return
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        st.markdown("---")
        
        # Region filter
        all_regions = ['All Regions'] + sorted(df['Region'].unique().tolist())
        selected_region = st.selectbox(
            "**Select Region**",
            all_regions,
            key="region_filter"
        )
        
        # Status filter
        all_statuses = ['All Statuses'] + sorted(df['Status'].dropna().unique().tolist())
        selected_status = st.multiselect(
            "**Select Status**",
            all_statuses,
            default=['All Statuses'],
            key="status_filter"
        )
        
        # Date range filter
        min_date = df['Incident Report Date from CRM'].min()
        max_date = df['Incident Report Date from CRM'].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input(
                "**Select Date Range**",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date,
                key="date_filter"
            )
        else:
            date_range = [None, None]
            st.warning("No valid dates available for filtering")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Parts", len(df))
        st.metric("Active Regions", df['Region'].nunique())
        st.metric("Data Period", f"{min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_region != 'All Regions':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if 'All Statuses' not in selected_status and selected_status:
        filtered_df = filtered_df[filtered_df['Status'].isin(selected_status)]
    
    if len(date_range) == 2 and all(date_range):
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Incident Report Date from CRM'] >= pd.Timestamp(start_date)) &
            (filtered_df['Incident Report Date from CRM'] <= pd.Timestamp(end_date))
        ]
    
    # Enhanced Key Performance Indicators Section
    st.markdown("## 📈 Executive Summary - Parts Performance")
    
    # Create 4 columns for main KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_incidents = len(filtered_df)
        st.markdown(create_metric_card(
            "TOTAL PARTS TRACKED", 
            f"{total_incidents:,}",
            "🔧"
        ), unsafe_allow_html=True)
    
    with col2:
        if filtered_df['Received in Days'].notna().any():
            avg_received_days = filtered_df['Received in Days'].mean()
            st.markdown(create_metric_card(
                "AVG PROCESSING DAYS", 
                f"{avg_received_days:.1f}",
                "⏱️"
            ), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card(
                "AVG PROCESSING DAYS", 
                "N/A",
                "⏱️"
            ), unsafe_allow_html=True)
    
    with col3:
        regions_covered = filtered_df['Region'].nunique()
        st.markdown(create_metric_card(
            "ACTIVE REGIONS", 
            f"{regions_covered}",
            "🌍"
        ), unsafe_allow_html=True)
    
    with col4:
        if 'Status' in filtered_df.columns and len(filtered_df) > 0:
            completed_count = (filtered_df['Status'] == 'Completed').sum()
            completion_rate = (completed_count / len(filtered_df)) * 100
            st.markdown(create_metric_card(
                "COMPLETION RATE", 
                f"{completion_rate:.1f}%",
                "✅"
            ), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card(
                "COMPLETION RATE", 
                "N/A",
                "✅"
            ), unsafe_allow_html=True)
    
    # Additional KPIs row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if 'Status' in filtered_df.columns:
            pending_count = (filtered_df['Status'] == 'Pending').sum()
            st.markdown(create_metric_card(
                "PENDING PARTS", 
                f"{pending_count:,}",
                "⏳"
            ), unsafe_allow_html=True)
    
    with col6:
        if 'Status' in filtered_df.columns:
            in_progress_count = (filtered_df['Status'] == 'In Progress').sum()
            st.markdown(create_metric_card(
                "IN PROGRESS", 
                f"{in_progress_count:,}",
                "🔄"
            ), unsafe_allow_html=True)
    
    with col7:
        if filtered_df['Received in Days'].notna().any():
            min_received_days = filtered_df['Received in Days'].min()
            st.markdown(create_metric_card(
                "MIN PROCESS DAYS", 
                f"{min_received_days:.1f}",
                "⚡"
            ), unsafe_allow_html=True)
    
    with col8:
        if filtered_df['Received in Days'].notna().any():
            max_received_days = filtered_df['Received in Days'].max()
            st.markdown(create_metric_card(
                "MAX PROCESS DAYS", 
                f"{max_received_days:.1f}",
                "📅"
            ), unsafe_allow_html=True)
    
    # Enhanced Charts Section
    st.markdown("---")
    st.markdown("## 📊 Parts Analytics & Distribution")
    
    # Row 1: Region Distribution and Status Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">🌍 Regional Parts Distribution</div>', 
                    unsafe_allow_html=True)
        
        if len(filtered_df) > 0:
            region_counts = filtered_df['Region'].value_counts()
            fig_region = px.bar(
                x=region_counts.index,
                y=region_counts.values,
                labels={'x': 'Region', 'y': 'Number of Parts'},
                color=region_counts.values,
                color_continuous_scale='blues',
                title="Parts Count by Region"
            )
            fig_region.update_layout(
                showlegend=False,
                height=450,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_region, use_container_width=True)
        else:
            st.info("📭 No data available for the selected filters")
    
    with col2:
        st.markdown('<div class="section-header">📈 Parts Status Overview</div>', 
                    unsafe_allow_html=True)
        
        if len(filtered_df) > 0 and 'Status' in filtered_df.columns:
            status_counts = filtered_df['Status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Parts Status Distribution"
            )
            fig_status.update_layout(height=450)
            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("📭 No status data available for the selected filters")
    
    # Row 2: Processing Time Analysis
    st.markdown('<div class="section-header">⏱️ Parts Processing Time Analysis</div>', 
                unsafe_allow_html=True)
    
    if len(filtered_df) > 0 and filtered_df['Received in Days'].notna().any():
        avg_days_by_region = filtered_df.groupby('Region')['Received in Days'].mean().sort_values(ascending=False)
        fig_avg_days = px.bar(
            x=avg_days_by_region.index,
            y=avg_days_by_region.values,
            labels={'x': 'Region', 'y': 'Average Processing Days'},
            color=avg_days_by_region.values,
            color_continuous_scale='viridis',
            title="Average Processing Time by Region (Days)"
        )
        fig_avg_days.update_layout(height=450, xaxis_tickangle=-45)
        st.plotly_chart(fig_avg_days, use_container_width=True)
    else:
        st.info("📭 No processing time data available for analysis")
    
    # Enhanced Data Summary Section
    st.markdown("---")
    st.markdown("## 🔍 Detailed Parts Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📋 Filter Summary</div>', 
                    unsafe_allow_html=True)
        summary_data = {
            'Metric': [
                'Total Parts Records', 
                'Active Regions', 
                'Date Range Start', 
                'Date Range End',
                'Selected Region',
                'Selected Statuses'
            ],
            'Value': [
                len(filtered_df),
                filtered_df['Region'].nunique(),
                filtered_df['Incident Report Date from CRM'].min().strftime('%Y-%m-%d') if not filtered_df.empty and pd.notna(filtered_df['Incident Report Date from CRM'].min()) else 'N/A',
                filtered_df['Incident Report Date from CRM'].max().strftime('%Y-%m-%d') if not filtered_df.empty and pd.notna(filtered_df['Incident Report Date from CRM'].max()) else 'N/A',
                selected_region,
                ', '.join(selected_status) if selected_status else 'All'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown('<div class="section-header">🏆 Top Regions by Parts Volume</div>', 
                    unsafe_allow_html=True)
        if len(filtered_df) > 0:
            top_regions = filtered_df['Region'].value_counts().head(8)
            fig_top = px.bar(
                x=top_regions.index,
                y=top_regions.values,
                labels={'x': 'Region', 'y': 'Parts Count'},
                color=top_regions.values,
                color_continuous_scale='thermal'
            )
            fig_top.update_layout(height=400)
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("📭 No data available")
    
    # Enhanced Data Export Section
    st.markdown("---")
    st.markdown("## 📥 Data Export & Management")
    
    if not filtered_df.empty:
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV Report",
                data=csv,
                file_name=f"awan_parts_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            # Summary statistics download
            summary_stats = filtered_df.describe().round(2)
            csv_summary = summary_stats.to_csv()
            st.download_button(
                label="📊 Download Summary Stats",
                data=csv_summary,
                file_name=f"awan_parts_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ No data available to export")
    
    # Enhanced Raw Data Preview
    with st.expander("🔍 View Detailed Parts Data"):
        if not filtered_df.empty:
            st.markdown(f"**Showing {len(filtered_df)} parts records**")
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.info("📭 No data available to display")

if __name__ == "__main__":
    main()