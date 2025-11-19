import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import re
import google.generativeai as genai
import time
import json
import os

# ==============================================
#            STREAMLIT DASHBOARD
#      Parts Intelligence Analytics System
# ==============================================

# Page configuration
st.set_page_config(
    page_title="HPE Parts Tracking Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
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
    .company-badge {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
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
    .example-query-btn {
        width: 100%;
        margin: 5px 0;
        padding: 12px;
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .example-query-btn:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .date-filter-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #bae6fd;
        margin-bottom: 1rem;
    }
    .data-table-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
#                1. ROBUST GOOGLE AI CLIENT
# ==============================================

class RobustGoogleAIClient:
    def __init__(self):
        self.configured = False
        self.rate_limited = False
        self.last_request_time = 0
        self.request_delay = 2  # 2 seconds between requests
        
    def configure(self, api_key):
        """Configure Google Generative AI with better error handling"""
        try:
            if not api_key:
                return False
                
            genai.configure(api_key=api_key)
            self.configured = True
            self.rate_limited = False
            return True
            
        except Exception as e:
            st.error(f"❌ Google AI Configuration Failed: {str(e)}")
            return False
    
    def generate_response(self, prompt, context_data):
        """Generate response with rate limiting and fallback"""
        if not self.configured:
            return self._get_enhanced_analytics(prompt, context_data)
        
        # Rate limiting protection
        current_time = time.time()
        if current_time - self.last_request_time < self.request_delay:
            time.sleep(self.request_delay)
        
        try:
            # Create context
            full_context = self._create_context(context_data)
            
            # Initialize model
            model = genai.GenerativeModel('gemini-pro')
            
            # Generate response
            response = model.generate_content(full_context + prompt)
            
            self.last_request_time = time.time()
            self.rate_limited = False
            
            return f"🤖 **AI Analysis:**\n\n{response.text}"
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific error types
            if "429" in error_msg or "resource exhausted" in error_msg.lower():
                self.rate_limited = True
                st.warning("⚠️ **AI Rate Limit Reached**: Using enhanced analytics instead")
                return self._get_enhanced_analytics(prompt, context_data)
            elif "503" in error_msg or "unavailable" in error_msg.lower():
                st.error("🔴 **AI Service Temporarily Unavailable**")
                return self._get_enhanced_analytics(prompt, context_data)
            else:
                return self._get_enhanced_analytics(prompt, context_data)
    
    def _get_enhanced_analytics(self, prompt, context_data):
        """Enhanced analytical responses without AI"""
        df = context_data
        prompt_lower = prompt.lower()
        
        # Comprehensive analytical responses
        if any(word in prompt_lower for word in ['analyze', 'performance', 'efficiency']):
            return self._generate_performance_analysis(df)
        elif any(word in prompt_lower for word in ['trend', 'pattern', 'time']):
            return self._generate_trend_analysis(df)
        elif any(word in prompt_lower for word in ['recommend', 'suggest', 'improve']):
            return self._generate_recommendations(df)
        elif any(word in prompt_lower for word in ['compare', 'region']):
            return self._generate_regional_comparison(df)
        elif any(word in prompt_lower for word in ['total', 'count', 'how many']):
            return self._generate_summary_analysis(df)
        else:
            return self._generate_general_insights(df, prompt)
    
    def _generate_performance_analysis(self, df):
        """Generate performance analysis"""
        total_parts = len(df)
        completed_parts = (df['Status'] == 'Completed').sum()
        completion_rate = (completed_parts / total_parts) * 100 if total_parts > 0 else 0
        pending_parts = (df['Status'] == 'Pending').sum()
        
        if df['Received in Days'].notna().any():
            avg_processing_time = df['Received in Days'].mean()
            processing_insight = f"{avg_processing_time:.1f} days"
        else:
            avg_processing_time = "N/A"
            processing_insight = "No data"
        
        analysis = f"""
📊 **Performance Analysis Report**

**Key Metrics:**
• Total Parts: **{total_parts:,}**
• Completed Parts: **{completed_parts:,}**
• Pending Parts: **{pending_parts:,}**
• Completion Rate: **{completion_rate:.1f}%**
• Avg Processing Time: **{processing_insight}**

**Performance Insights:**
1. **Efficiency Score**: {min(100, completion_rate + (20 if avg_processing_time != "N/A" and avg_processing_time < 5 else 0))}/100
2. **Bottleneck Identification**: {self._identify_bottlenecks(df)}
3. **Improvement Opportunities**: {self._suggest_improvements(df)}

**Recommendations:**
• Focus on reducing {pending_parts} pending parts
• Optimize regional resource allocation
• Implement quality control measures
• Target completion rate of 85%+
        """
        return f"🔍 **Enhanced Analytics:**\n\n{analysis}"
    
    def _generate_trend_analysis(self, df):
        """Generate trend analysis"""
        if df['Incident Report Date from CRM'].notna().any():
            monthly_trend = df.set_index('Incident Report Date from CRM').resample('M').size()
            if len(monthly_trend) > 1:
                trend_direction = "increasing" if monthly_trend.iloc[-1] > monthly_trend.iloc[0] else "decreasing"
                trend_strength = abs(monthly_trend.iloc[-1] - monthly_trend.iloc[0]) / monthly_trend.iloc[0] * 100
            else:
                trend_direction = "stable"
                trend_strength = 0
        else:
            trend_direction = "insufficient data"
            trend_strength = 0
        
        regional_dist = df['Region'].value_counts()
        top_region = regional_dist.index[0] if len(regional_dist) > 0 else "N/A"
        top_region_count = regional_dist.iloc[0] if len(regional_dist) > 0 else 0
        
        analysis = f"""
📈 **Trend Analysis Report**

**Current Trends:**
• Overall Activity: **{trend_direction}** ({trend_strength:.1f}% change)
• Top Performing Region: **{top_region}** ({top_region_count} parts)
• Regional Distribution: **{len(regional_dist)}** active regions

**Pattern Insights:**
1. **Seasonal Patterns**: {self._detect_seasonal_patterns(df)}
2. **Regional Hotspots**: {self._identify_regional_hotspots(df)}
3. **Processing Trends**: {self._analyze_processing_trends(df)}

**Forecast:**
• Expected completion rate improvement: **5-10%** with current optimizations
• Potential bottleneck regions: {self._identify_potential_bottlenecks(df)}
• Recommended focus areas: {self._get_improvement_focus(df)}
        """
        return f"🔍 **Enhanced Analytics:**\n\n{analysis}"
    
    def _generate_recommendations(self, df):
        """Generate data-driven recommendations"""
        total_parts = len(df)
        pending_parts = (df['Status'] == 'Pending').sum()
        completed_parts = (df['Status'] == 'Completed').sum()
        
        if df['Region'].nunique() > 0:
            regional_imbalance = df['Region'].value_counts().std() / df['Region'].value_counts().mean()
        else:
            regional_imbalance = 0
        
        rejected_parts = df['Status'].str.contains('Rejected', na=False).sum()
        
        recommendations = f"""
💡 **Strategic Recommendations**

**Immediate Actions (1-2 weeks):**
1. **Address {pending_parts} pending parts** - Assign dedicated resources
2. **Regional Balance** - Current imbalance score: {regional_imbalance:.2f}
3. **Quality Control** - Review {rejected_parts} rejected parts

**Medium-term (1-3 months):**
1. **Process Optimization** - Target 20% reduction in processing time
2. **Resource Reallocation** - Based on regional demand patterns
3. **Supplier Management** - Address part quality issues

**Long-term Strategy:**
1. **Predictive Maintenance** - Implement forecasting systems
2. **Inventory Optimization** - Reduce unused parts stock
3. **Performance Monitoring** - Real-time KPI tracking

**Success Metrics:**
• Reduce pending parts by 50% in 30 days
• Increase completion rate to 85%+
• Improve regional balance score below 0.5
        """
        return f"🔍 **Enhanced Analytics:**\n\n{recommendations}"
    
    def _generate_regional_comparison(self, df):
        """Generate regional comparison"""
        if df['Region'].nunique() > 0:
            regional_stats = df.groupby('Region').agg({
                'Status': lambda x: (x == 'Completed').mean() * 100,
                'Received in Days': 'mean'
            }).round(1)
            
            best_region = regional_stats['Status'].idxmax() if len(regional_stats) > 0 else "N/A"
            worst_region = regional_stats['Status'].idxmin() if len(regional_stats) > 0 else "N/A"
            
            fastest_region = regional_stats['Received in Days'].idxmin() if regional_stats['Received in Days'].notna().any() else "N/A"
            slowest_region = regional_stats['Received in Days'].idxmax() if regional_stats['Received in Days'].notna().any() else "N/A"
        else:
            best_region = worst_region = fastest_region = slowest_region = "N/A"
        
        comparison = f"""
🌍 **Regional Performance Comparison**

**Top Performers:**
• **{best_region}**: {regional_stats.loc[best_region, 'Status'] if best_region != 'N/A' else 'N/A'}% completion rate
• **{fastest_region}**: Fastest processing time

**Areas Needing Improvement:**
• **{worst_region}**: {regional_stats.loc[worst_region, 'Status'] if worst_region != 'N/A' else 'N/A'}% completion rate
• **{slowest_region}**: Slowest processing time

**Regional Insights:**
• Performance variance: {regional_stats['Status'].std() if best_region != 'N/A' else 0:.1f}%
• Average completion: {regional_stats['Status'].mean() if best_region != 'N/A' else 0:.1f}%

**Recommendations:**
• Share best practices from {best_region}
• Investigate challenges in {worst_region}
• Implement regional performance benchmarks
        """
        return f"🔍 **Enhanced Analytics:**\n\n{comparison}"
    
    def _generate_summary_analysis(self, df):
        """Generate summary analysis"""
        total_parts = len(df)
        status_summary = df['Status'].value_counts().to_dict()
        regional_summary = df['Region'].value_counts().head(3).to_dict()
        
        if df['Received in Days'].notna().any():
            avg_processing = df['Received in Days'].mean()
            processing_insight = f"{avg_processing:.1f} days average"
        else:
            processing_insight = "No processing time data"
        
        summary = f"""
📋 **Comprehensive Summary Report**

**Overview:**
• Total Parts: **{total_parts:,}**
• Active Regions: **{df['Region'].nunique()}**
• Processing Time: **{processing_insight}**

**Status Distribution:**
{chr(10).join([f"• **{status}**: {count} parts ({count/total_parts*100:.1f}%)" for status, count in list(status_summary.items())[:5]])}

**Top Regions:**
{chr(10).join([f"• **{region}**: {count} parts" for region, count in list(regional_summary.items())[:3]])}

**Key Insights:**
1. Operational Scale: **{self._get_scale_indicator(total_parts)}**
2. Efficiency Level: **{self._get_efficiency_potential(df)}**
3. Primary Focus: **{self._get_improvement_focus(df)}**

**Quick Actions:**
• Monitor {list(status_summary.keys())[0] if status_summary else 'N/A'} parts closely
• Review regional distribution patterns
• Analyze processing time efficiency
        """
        return f"🔍 **Enhanced Analytics:**\n\n{summary}"
    
    def _generate_general_insights(self, df, prompt):
        """Generate general insights for other queries"""
        total_parts = len(df)
        status_summary = df['Status'].value_counts().to_dict()
        regional_summary = df['Region'].value_counts().head(3).to_dict()
        
        insights = f"""
🔍 **Data Insights for: "{prompt}"**

**Quick Overview:**
• Total Parts: **{total_parts:,}**
• Status Distribution: {', '.join([f'{k}: {v}' for k, v in list(status_summary.items())[:3]])}
• Top Regions: {', '.join([f'{k}: {v}' for k, v in regional_summary.items()])}

**Key Observations:**
1. Operational Scale: **{self._get_scale_indicator(total_parts)}**
2. Efficiency Potential: **{self._get_efficiency_potential(df)}**
3. Improvement Focus: **{self._get_improvement_focus(df)}**

**Suggested Next Questions:**
• "Show me regional performance comparison"
• "Analyze processing time trends" 
• "Recommend efficiency improvements"
• "What's our completion rate breakdown?"
        """
        return f"🔍 **Enhanced Analytics:**\n\n{insights}"
    
    # Helper methods for enhanced analytics
    def _identify_bottlenecks(self, df):
        pending = (df['Status'] == 'Pending').sum()
        total = len(df)
        if total == 0:
            return "No data available"
        if pending / total > 0.3:
            return "High pending parts indicate processing delays"
        elif pending / total > 0.1:
            return "Moderate pending parts - monitor closely"
        else:
            return "Low pending parts - good workflow"
    
    def _suggest_improvements(self, df):
        suggestions = []
        if (df['Status'] == 'Pending').sum() > 0:
            suggestions.append("Reduce pending parts through faster processing")
        if df['Region'].nunique() > 1:
            suggestions.append("Optimize regional resource allocation")
        if (df['Status'].str.contains('Rejected', na=False)).any():
            suggestions.append("Improve quality control measures")
        if df['Received in Days'].mean() > 7:
            suggestions.append("Accelerate processing timeline")
        return "; ".join(suggestions[:3])
    
    def _detect_seasonal_patterns(self, df):
        return "Check monthly trends in dashboard for detailed pattern analysis"
    
    def _identify_regional_hotspots(self, df):
        regional_counts = df['Region'].value_counts()
        if len(regional_counts) > 0:
            return f"High activity in {regional_counts.index[0]} ({regional_counts.iloc[0]} parts)"
        return "No regional data"
    
    def _analyze_processing_trends(self, df):
        if df['Received in Days'].notna().any():
            avg_time = df['Received in Days'].mean()
            return f"Average processing: {avg_time:.1f} days"
        return "Processing time data available in dashboard"
    
    def _identify_potential_bottlenecks(self, df):
        regional_stats = df.groupby('Region').size()
        if len(regional_stats) > 0:
            return regional_stats.idxmax()
        return "Analyze regional distribution"
    
    def _get_scale_indicator(self, total_parts):
        if total_parts > 1000:
            return "Large-scale operations"
        elif total_parts > 500:
            return "Medium-scale operations"
        else:
            return "Small-scale operations"
    
    def _get_efficiency_potential(self, df):
        if len(df) == 0:
            return "No data"
        completion_rate = (df['Status'] == 'Completed').mean() * 100
        if completion_rate > 80:
            return "High efficiency"
        elif completion_rate > 60:
            return "Moderate efficiency - room for improvement"
        else:
            return "Significant improvement potential"
    
    def _get_improvement_focus(self, df):
        if len(df) == 0:
            return "Data analysis"
        if (df['Status'] == 'Pending').sum() > 0:
            return "Reduce pending parts"
        elif df['Received in Days'].mean() > 7:
            return "Accelerate processing time"
        elif df['Region'].nunique() > 1:
            return "Optimize regional balance"
        else:
            return "Maintain current performance"
    
    def _create_context(self, df):
        """Create basic context for AI"""
        try:
            total_parts = len(df)
            regions = df['Region'].nunique()
            status_counts = df['Status'].value_counts().to_dict()
            
            context = f"""
            Parts Data Analysis:
            - Total Parts: {total_parts}
            - Regions: {regions}
            - Status: {status_counts}
            """
            return context
        except:
            return "Analyze this parts data."

# ==============================================
#                2. DATA LOADING
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

# ==============================================
#                3. HYBRID CHATBOT
# ==============================================

class HybridChatbot:
    def __init__(self, df):
        self.df = df
        self.ai_client = RobustGoogleAIClient()
        
    def process_query(self, query, filtered_df, api_key=None):
        """Process queries with enhanced fallback"""
        # Configure AI if API key provided
        if api_key and not self.ai_client.configured:
            self.ai_client.configure(api_key)
        
        # Always use enhanced analytics (more reliable)
        return self.ai_client._get_enhanced_analytics(query, filtered_df)

# ==============================================
#               4. DASHBOARD LAYOUT
# ==============================================

def main():
    # Enhanced Header Section
    st.markdown('<div class="company-badge">PARTS INTELLIGENCE ANALYTICS SYSTEM</div>', 
                unsafe_allow_html=True)
    
    st.markdown(
        '<div class="main-header">🔧 Advanced Parts Intelligence Dashboard</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<div class="sub-header">Enterprise Analytics • Multi-Regional Tracking • AI-Powered Insights</div>', 
        unsafe_allow_html=True
    )
    
    # Load data
    with st.spinner('🔄 Loading Parts Intelligence Data...'):
        df = load_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check the data source.")
        return
    
    if df.empty:
        st.warning("⚠️ No data available with the current filters.")
        return
    
    # Initialize chatbot
    chatbot = HybridChatbot(df)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        st.markdown("---")
        
        # Date Range Filter
        st.markdown("### 📅 Date Range Filter")
        if df['Incident Report Date from CRM'].notna().any():
            min_date = df['Incident Report Date from CRM'].min().date()
            max_date = df['Incident Report Date from CRM'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="end_date"
                )
        else:
            st.info("No date data available")
            start_date = end_date = None
        
        st.markdown("---")
        
        # Google AI Configuration (Optional)
        st.markdown("### 🤖 Google AI Setup (Optional)")
        
        api_key = st.text_input(
            "Enter Google AI API Key:",
            type="password",
            help="Get free API key from: https://aistudio.google.com/app/apikey",
            key="api_key_input"
        )
        
        if api_key:
            if chatbot.ai_client.configure(api_key):
                st.success("✅ Google AI Connected!")
            else:
                st.error("❌ Invalid API Key")
        
        st.info("💡 **No API key needed!** Advanced analytics work without it!")
        
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
        
        # AI Mode Toggle
        st.markdown("---")
        st.markdown("### 🧠 Analysis Mode")
        ai_mode = st.toggle("Enable Google AI (Optional)", value=False, 
                           help="Use Google Gemini for additional insights (requires API key)",
                           key="ai_mode_toggle")
        
        st.markdown("---")
        st.markdown("### 💡 Quick Questions")
        
        # Quick action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Total", key="btn_total", use_container_width=True):
                st.session_state.chat_input = "How many total parts do we have?"
                st.rerun()
        with col2:
            if st.button("🌍 Regions", key="btn_regions", use_container_width=True):
                st.session_state.chat_input = "Analyze regional distribution and performance"
                st.rerun()
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("⏱️ Efficiency", key="btn_efficiency", use_container_width=True):
                st.session_state.chat_input = "What's our processing efficiency and how can we improve it?"
                st.rerun()
        with col4:
            if st.button("📈 Trends", key="btn_trends", use_container_width=True):
                st.session_state.chat_input = "Show me performance trends and patterns over time"
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 System Status")
        st.success("✅ **Enhanced Analytics**: Always Active")
        if ai_mode and chatbot.ai_client.configured:
            st.success("✅ **Google AI**: Connected")
        elif ai_mode:
            st.warning("⚠️ **Google AI**: Needs API Key")
        else:
            st.info("ℹ️ **Google AI**: Disabled")
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply date filter
    if start_date and end_date and df['Incident Report Date from CRM'].notna().any():
        filtered_df = filtered_df[
            (filtered_df['Incident Report Date from CRM'].dt.date >= start_date) & 
            (filtered_df['Incident Report Date from CRM'].dt.date <= end_date)
        ]
    
    if selected_region != 'All Regions':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if 'All Statuses' not in selected_status and selected_status:
        filtered_df = filtered_df[filtered_df['Status'].isin(selected_status)]
    
    
    # Main layout with tabs
    tab1, tab2 = st.tabs(["📊 Dashboard Analytics", "🤖 AI Chat Assistant"])
    
    with tab1:
        display_dashboard(filtered_df, selected_region, selected_status, start_date, end_date)
    
    with tab2:
        display_chat_interface(chatbot, filtered_df, ai_mode, api_key)

def display_dashboard(filtered_df, selected_region, selected_status, start_date, end_date):
    """Display the main dashboard analytics"""
    
    # Key Performance Indicators
    st.markdown("## 📈 Executive Summary - Parts Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_parts = len(filtered_df)
        st.metric("Total Parts Tracked", f"{total_parts:,}")
    
    with col2:
        if filtered_df['Received in Days'].notna().any():
            avg_processing_days = filtered_df['Received in Days'].mean()
            st.metric("Avg Processing Days", f"{avg_processing_days:.1f}")
        else:
            st.metric("Avg Processing Days", "N/A")
    
    with col3:
        regions_covered = filtered_df['Region'].nunique()
        st.metric("Active Regions", f"{regions_covered}")
    
    with col4:
        if 'Status' in filtered_df.columns and len(filtered_df) > 0:
            completed_count = (filtered_df['Status'] == 'Completed').sum()
            completion_rate = (completed_count / len(filtered_df)) * 100
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        else:
            st.metric("Completion Rate", "N/A")
    
    # Additional Metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        pending_count = (filtered_df['Status'] == 'Pending').sum()
        st.metric("Pending Parts", f"{pending_count:,}")
    
    with col6:
        in_progress_count = (filtered_df['Status'] == 'In Progress').sum()
        st.metric("In Progress", f"{in_progress_count:,}")
    
    with col7:
        rejected_count = filtered_df['Status'].str.contains('Rejected', na=False).sum()
        st.metric("Rejected Parts", f"{rejected_count:,}")
    
    with col8:
        unused_count = (filtered_df['Status'] == 'Unused').sum()
        st.metric("Unused Parts", f"{unused_count:,}")
    
    # NEW: Display Filtered Data Table
    st.markdown("---")
    st.markdown("### 📋 Filtered Parts Data")
    
    if len(filtered_df) > 0:
        # Display data table with customer information
        st.markdown(f"**Showing {len(filtered_df)} records**")
        
        # Create a copy for display with better column names
        display_df = filtered_df.copy()
        
        # Select important columns to display (you can modify this based on your actual columns)
        important_columns = [
            'Region', 'Status', 'Incident Report Date from CRM',
            'Customer Name', 'TAG No.', 'Part Description'
        ]
        
        # Filter to only show columns that exist in the dataframe
        available_columns = [col for col in important_columns if col in display_df.columns]
        display_df = display_df[available_columns]
        
        # Display the dataframe
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_parts_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.info(f"💡 **Tip**: The table shows {len(filtered_df)} filtered records. Use the download button to export the complete dataset.")
    
    else:
        st.warning("No data available with the current filters.")
    
    # Charts
    st.markdown("---")
    st.markdown("## 📊 Parts Analytics & Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌍 Regional Parts Distribution")
        if len(filtered_df) > 0:
            region_counts = filtered_df['Region'].value_counts()
            fig_region = px.bar(
                x=region_counts.index,
                y=region_counts.values,
                labels={'x': 'Region', 'y': 'Number of Parts'},
                color=region_counts.values,
                color_continuous_scale='blues',
                title="Parts Distribution Across Regions"
            )
            fig_region.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_region, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    with col2:
        st.markdown("### 📈 Parts Status Overview")
        if len(filtered_df) > 0 and 'Status' in filtered_df.columns:
            status_counts = filtered_df['Status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Parts Status Distribution"
            )
            fig_status.update_layout(height=400)
            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("No status data available")
    
    # Processing Time Analysis
    st.markdown("### ⏱️ Processing Time Analysis")
    if len(filtered_df) > 0 and filtered_df['Received in Days'].notna().any():
        avg_days_by_region = filtered_df.groupby('Region')['Received in Days'].mean().sort_values(ascending=False)
        fig_avg_days = px.bar(
            x=avg_days_by_region.index,
            y=avg_days_by_region.values,
            labels={'x': 'Region', 'y': 'Average Processing Days'},
            color=avg_days_by_region.values,
            color_continuous_scale='viridis',
            title="Average Processing Time by Region"
        )
        fig_avg_days.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_avg_days, use_container_width=True)
    else:
        st.info("No processing time data available for analysis")
    
    # Time Series Analysis
    st.markdown("### 📅 Monthly Parts Activity")
    if len(filtered_df) > 0 and filtered_df['Incident Report Date from CRM'].notna().any():
        monthly_trend = filtered_df.set_index('Incident Report Date from CRM').resample('M').size()
        fig_trend = px.line(
            x=monthly_trend.index,
            y=monthly_trend.values,
            labels={'x': 'Month', 'y': 'Number of Parts'},
            title="Monthly Parts Activity Trend"
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No date data available for trend analysis")

def display_chat_interface(chatbot, filtered_df, ai_mode, api_key):
    """Display the chat interface"""
    
    st.markdown("## 🤖 Parts Intelligence AI Assistant")
    
    # System Status
    if ai_mode and chatbot.ai_client.configured:
        st.success("🧠 **Advanced Mode**: Google AI + Enhanced Analytics")
    else:
        st.success("🔍 **Enhanced Analytics**: Always Active - No API Required")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm your Parts Intelligence Assistant. I can help you analyze parts data, provide insights, and suggest improvements. Ask me anything about parts performance, regional distribution, or efficiency metrics!"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initialize chat input in session state if not exists
    if 'chat_input' not in st.session_state:
        st.session_state.chat_input = ""
    
    # Chat input
    chat_input = st.chat_input(
        "Ask about parts data, trends, efficiency...",
        key="chat_input_widget"
    )
    
    # Check if we have a pending query from buttons
    if st.session_state.chat_input:
        prompt = st.session_state.chat_input
        st.session_state.chat_input = ""  # Clear after reading
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing data..."):
                if ai_mode and chatbot.ai_client.configured:
                    response = chatbot.ai_client.generate_response(prompt, filtered_df)
                else:
                    response = chatbot.process_query(prompt, filtered_df, api_key)
            
            st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to clear the input and show new messages
        st.rerun()
    
    elif chat_input:
        # Handle direct chat input
        prompt = chat_input
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing data..."):
                if ai_mode and chatbot.ai_client.configured:
                    response = chatbot.ai_client.generate_response(prompt, filtered_df)
                else:
                    response = chatbot.process_query(prompt, filtered_df, api_key)
            
            st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ Clear Chat History", key="clear_chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 Hello! I'm your Parts Intelligence Assistant. How can I help you analyze your parts data today?"}
            ]
            st.rerun()
    
    # Example queries with proper styling and functionality
    st.markdown("---")
    st.markdown("### 💡 Example Queries:")
    
    # Use columns for better layout
    examples_col1, examples_col2, examples_col3 = st.columns(3)
    
    with examples_col1:
        if st.button("🏆 Performance Analysis", key="example1", use_container_width=True):
            st.session_state.chat_input = "Analyze our overall parts processing performance and suggest specific improvements"
            st.rerun()
    
    with examples_col2:
        if st.button("📈 Trend Insights", key="example2", use_container_width=True):
            st.session_state.chat_input = "Identify trends and patterns in our parts processing timeline and suggest optimization strategies"
            st.rerun()
    
    with examples_col3:
        if st.button("🔍 Efficiency Audit", key="example3", use_container_width=True):
            st.session_state.chat_input = "Conduct a comprehensive efficiency audit and recommend specific optimization strategies for parts processing"
            st.rerun()
    
    # Second row of example queries
    examples_col4, examples_col5, examples_col6 = st.columns(3)
    
    with examples_col4:
        if st.button("🌍 Regional Compare", key="example4", use_container_width=True):
            st.session_state.chat_input = "Compare regional performance and identify best practices"
            st.rerun()
    
    with examples_col5:
        if st.button("📋 Summary Report", key="example5", use_container_width=True):
            st.session_state.chat_input = "Provide a comprehensive summary of our parts operations"
            st.rerun()
    
    with examples_col6:
        if st.button("💡 Improvement Ideas", key="example6", use_container_width=True):
            st.session_state.chat_input = "Suggest specific improvements for our parts management process"
            st.rerun()

if __name__ == "__main__":
    main()
