import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Air France Customer Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .positive {
        color: #10B981;
        font-weight: bold;
    }
    .negative {
        color: #EF4444;
        font-weight: bold;
    }
    .neutral {
        color: #F59E0B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        # Load sentiment analysis results
        df = pd.read_csv('airfrance_topics_analysis.csv')
        
        # Load business impact report
        with open('airfrance_final_report.json', 'r') as f:
            report = json.load(f)
        
        # Load summary statistics
        with open('sentiment_analysis_summary.json', 'r') as f:
            sentiment_summary = json.load(f)
            
        return df, report, sentiment_summary
    except Exception as e:
        st.error(f"Data loading error: {e}")
        # Create sample data if files don't exist
        st.warning("Creating sample data for demonstration...")
        
        # Create sample dataframe
        sample_data = {
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'text': [
                "Air France customer service is terrible! Been on hold for 2 hours.",
                "Trying to change my Nigeria flight due to visa delay. Air France wants huge fees.",
                "Every time I call Air France, I get a different agent and have to explain everything again.",
                "Air France agent couldn't understand my English properly.",
                "Great service from Air France today! Agent was very helpful."
            ] * 20,
            'sentiment': ['Negative', 'Negative', 'Negative', 'Negative', 'Positive'] * 20,
            'sentiment_score': [0.95, 0.92, 0.89, 0.91, 0.88] * 20,
            'topic': ['Call Center', 'Visa Issues', 'Agent Continuity', 'Language Barriers', 'Positive'] * 20,
            'engagement_score': np.random.randint(10, 100, 100)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Create sample reports
        report = {
            'total_complaints': 100,
            'total_potential_loss': 5000000,
            'total_recoverable_revenue': 1500000,
            'estimated_annual_roi': 375000,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'top_issues': [
                {'topic': 'Visa Issues', 'complaints': 25, 'negative_rate': 0.85, 'recoverable_revenue': 450000},
                {'topic': 'Agent Continuity', 'complaints': 20, 'negative_rate': 0.80, 'recoverable_revenue': 400000},
                {'topic': 'Language Barriers', 'complaints': 15, 'negative_rate': 0.75, 'recoverable_revenue': 300000}
            ],
            'girlfriend_scenario_impact': {
                'topics': ['Visa Issues', 'Agent Continuity', 'Language Barriers'],
                'recoverable_revenue': 1150000
            }
        }
        
        sentiment_summary = {
            'negative_percentage': 65.0
        }
        
        return df, report, sentiment_summary

# Load data
df, report, sentiment_summary = load_data()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Air_France_Logo.svg/1280px-Air_France_Logo.svg.png", 
             width=200, use_container_width=True)
    
    st.title("✈️ Air France CX Intelligence")
    st.markdown("**Data-Driven Customer Experience Analysis**")
    st.markdown("---")
    
    # Date range selector
    if df is not None and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Topic filter
    if df is not None and 'topic' in df.columns:
        topics = ['All Topics'] + sorted(df['topic'].unique().tolist())
        selected_topic = st.selectbox("Filter by Topic", topics)
    
    st.markdown("---")
    st.info(f"**Analysis Period:** {report['total_complaints'] if report else 'N/A'} complaints analyzed")

# Main content
if df is None or report is None:
    st.warning("Please run the data analysis pipeline first.")
    st.stop()

# Header
st.markdown('<h1 class="main-header">✈️ Air France Customer Service Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Transforming Customer Complaints into Business Opportunities")

# Key Metrics Row
st.markdown('<h3 class="sub-header">📊 Key Performance Metrics</h3>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Total Complaints</h4>
        <h2>{report['total_complaints']:,}</h2>
        <p>Customer feedback analyzed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    negative_rate = sentiment_summary.get('negative_percentage', 0)
    st.markdown(f"""
    <div class="metric-card">
        <h4>Negative Sentiment</h4>
        <h2 class="negative">{negative_rate:.1f}%</h2>
        <p>Customer dissatisfaction rate</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Revenue at Risk</h4>
        <h2>€{report['total_potential_loss']:,.0f}</h2>
        <p>Potential customer churn impact</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Recoverable Revenue</h4>
        <h2 class="positive">€{report['total_recoverable_revenue']:,.0f}</h2>
        <p>Through service improvements</p>
    </div>
    """, unsafe_allow_html=True)

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview", 
    "🔍 Deep Analysis", 
    "💰 Business Impact", 
    "🚀 Recommendations"
])

with tab1:
    st.markdown('<h3 class="sub-header">Overall Sentiment & Topic Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            fig1 = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.3,
                marker_colors=['#EF4444', '#F59E0B', '#10B981']
            )])
            fig1.update_layout(title_text="Sentiment Distribution", showlegend=True)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Topic Distribution
        if 'topic' in df.columns:
            topic_counts = df['topic'].value_counts().head(10)
            fig2 = go.Figure(data=[go.Bar(
                x=topic_counts.values,
                y=topic_counts.index,
                orientation='h',
                marker_color='#3B82F6'
            )])
            fig2.update_layout(
                title_text="Top 10 Complaint Topics",
                xaxis_title="Number of Complaints",
                yaxis_title="Topic"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Monthly Trends
    st.markdown('<h3 class="sub-header">📅 Monthly Complaint Trends</h3>', unsafe_allow_html=True)
    
    if 'date' in df.columns:
        df['month'] = df['date'].dt.strftime('%Y-%m')
        monthly_data = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
        
        fig3 = go.Figure()
        colors = {'Negative': '#EF4444', 'Neutral': '#F59E0B', 'Positive': '#10B981'}
        
        for sentiment in ['Negative', 'Neutral', 'Positive']:
            if sentiment in monthly_data.columns:
                fig3.add_trace(go.Bar(
                    x=monthly_data.index,
                    y=monthly_data[sentiment],
                    name=sentiment,
                    marker_color=colors[sentiment]
                ))
        
        fig3.update_layout(
            barmode='stack',
            title="Monthly Complaint Volume by Sentiment",
            xaxis_title="Month",
            yaxis_title="Number of Complaints",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown('<h3 class="sub-header">Deep Dive into Complaint Patterns</h3>', unsafe_allow_html=True)
    
    # Topic-Sentiment Analysis
    if 'topic' in df.columns and 'sentiment' in df.columns:
        # Create sentiment by topic heatmap
        sentiment_by_topic = pd.crosstab(df['topic'], df['sentiment'], normalize='index') * 100
        
        # Filter to topics with sufficient data
        topic_counts = df['topic'].value_counts()
        significant_topics = topic_counts[topic_counts >= 3].index
        sentiment_by_topic = sentiment_by_topic.loc[significant_topics]
        
        # Sort by negative sentiment
        if 'Negative' in sentiment_by_topic.columns:
            sentiment_by_topic = sentiment_by_topic.sort_values('Negative', ascending=False)
        
        # Create heatmap
        fig4 = go.Figure(data=go.Heatmap(
            z=sentiment_by_topic.values,
            x=sentiment_by_topic.columns,
            y=sentiment_by_topic.index,
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=100,
            text=np.round(sentiment_by_topic.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10}
        ))
        
        fig4.update_layout(
            title="Sentiment Distribution by Topic (%)",
            xaxis_title="Sentiment",
            yaxis_title="Topic",
            height=500
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Word Cloud and Sample Complaints
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔥 Top Negative Complaints")
        if 'sentiment_score' in df.columns:
            top_negative = df[df['sentiment'] == 'Negative'].nlargest(5, 'sentiment_score')
            
            for idx, row in top_negative.iterrows():
                with st.expander(f"Complaint #{idx+1} - Score: {row['sentiment_score']:.3f}"):
                    st.write(f"**Topic:** {row['topic']}")
                    st.write(f"**Text:** {row['text']}")
                    if 'date' in row:
                        st.write(f"**Date:** {row['date'].date()}")
    
    with col2:
        st.markdown("#### 📊 Topic Engagement Analysis")
        if 'engagement_score' in df.columns and 'topic' in df.columns:
            engagement_by_topic = df.groupby('topic')['engagement_score'].mean().sort_values(ascending=False)
            
            fig5 = go.Figure(data=[go.Bar(
                x=engagement_by_topic.values,
                y=engagement_by_topic.index,
                orientation='h',
                marker_color='#8B5CF6'
            )])
            fig5.update_layout(
                title="Average Engagement by Topic",
                xaxis_title="Average Engagement Score",
                yaxis_title="Topic",
                height=300
            )
            st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.markdown('<h3 class="sub-header">💰 Business Impact Analysis</h3>', unsafe_allow_html=True)
    
    # Financial Impact Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue impact chart
        impact_data = {
            'Category': ['Potential Loss', 'Recoverable Revenue', 'Annual ROI'],
            'Amount (€)': [
                report['total_potential_loss'],
                report['total_recoverable_revenue'],
                report.get('estimated_annual_roi', report['total_recoverable_revenue'] * 0.25)
            ]
        }
        
        impact_df = pd.DataFrame(impact_data)
        
        fig6 = go.Figure(data=[go.Bar(
            x=impact_df['Category'],
            y=impact_df['Amount (€)'],
            marker_color=['#EF4444', '#10B981', '#3B82F6']
        )])
        
        fig6.update_layout(
            title="Financial Impact Breakdown",
            yaxis_title="Amount (€)",
            height=400
        )
        
        # Format y-axis with commas
        fig6.update_yaxes(tickformat=",")
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        # Your girlfriend's scenario impact
        st.markdown("#### 🎯 Your Specific Scenario Impact")
        
        girlfriend_impact = report.get('girlfriend_scenario_impact', {
            'topics': ['Visa Issues', 'Agent Continuity', 'Language Barriers'],
            'recoverable_revenue': report['total_recoverable_revenue'] * 0.7
        })
        
        st.info(f"""
        **Topics Affecting Your Experience:**
        {', '.join(girlfriend_impact['topics'])}
        
        **Recoverable Revenue from These Issues:**
        €{girlfriend_impact['recoverable_revenue']:,.0f}
        
        **Monthly Impact:**
        €{girlfriend_impact['recoverable_revenue']/12:,.0f}
        """)
    
    # Top Issues Business Impact
    st.markdown("#### 🔴 Top 5 High-Impact Issues")
    
    issues_data = []
    for issue in report.get('top_issues', []):
        issues_data.append({
            'Issue': issue['topic'],
            'Complaints': issue['complaints'],
            'Negative Rate': f"{issue['negative_rate']*100:.1f}%",
            'Recoverable Revenue': f"€{issue['recoverable_revenue']:,.0f}"
        })
    
    # If no issues data, create sample
    if not issues_data:
        issues_data = [
            {'Issue': 'Visa Issues', 'Complaints': 25, 'Negative Rate': '85.0%', 'Recoverable Revenue': '€450,000'},
            {'Issue': 'Agent Continuity', 'Complaints': 20, 'Negative Rate': '80.0%', 'Recoverable Revenue': '€400,000'},
            {'Issue': 'Language Barriers', 'Complaints': 15, 'Negative Rate': '75.0%', 'Recoverable Revenue': '€300,000'}
        ]
    
    issues_df = pd.DataFrame(issues_data)
    st.dataframe(issues_df, use_container_width=True, hide_index=True)

with tab4:
    st.markdown('<h3 class="sub-header">🚀 Data-Driven Recommendations</h3>', unsafe_allow_html=True)
    
    # Recommendations based on top issues
    recommendations = {
        'Visa & Rebooking Issues': {
            'Priority': 'High',
            'Problem': 'Complex visa-related rebooking with high fees causing customer frustration',
            'Solution': 'Create specialized visa rebooking workflow with pre-approved alternatives',
            'Impact': 'Reduce rebooking time by 70%, increase customer satisfaction scores',
            'Timeline': '2-4 months',
            'ROI': f"€{report.get('top_issues', [{}])[0].get('recoverable_revenue', 450000) * 0.3:,.0f}/year"
        },
        'Agent Continuity Problems': {
            'Priority': 'Very High',
            'Problem': 'Customers repeat information to multiple agents, causing frustration',
            'Solution': 'Implement case tracking system with "continue with same agent" option',
            'Impact': 'Reduce call duration by 30%, improve first-call resolution rate',
            'Timeline': '3-6 months',
            'ROI': f"€{report.get('top_issues', [{}])[1].get('recoverable_revenue', 400000) * 0.25:,.0f}/year"
        },
        'Language Barriers': {
            'Priority': 'Medium',
            'Problem': 'Communication issues affecting service quality',
            'Solution': 'Language proficiency certification + real-time translation tools',
            'Impact': 'Reduce miscommunication by 50%, improve CSAT scores',
            'Timeline': '6-12 months',
            'ROI': f"€{report.get('top_issues', [{}])[2].get('recoverable_revenue', 300000) * 0.2:,.0f}/year"
        }
    }
    
    for topic, rec in recommendations.items():
        with st.expander(f"✅ {topic} (Priority: {rec['Priority']})", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Problem:** {rec['Problem']}")
                st.markdown(f"**Solution:** {rec['Solution']}")
            
            with col2:
                st.markdown(f"**Expected Impact:** {rec['Impact']}")
                st.markdown(f"**Timeline:** {rec['Timeline']}")
                st.markdown(f"**Estimated ROI:** {rec['ROI']}")
    
    # Implementation Roadmap
    st.markdown("#### 📅 Implementation Roadmap")
    
    roadmap_data = {
        'Phase': ['Phase 1 (0-3 months)', 'Phase 2 (3-6 months)', 'Phase 3 (6-12 months)'],
        'Focus': ['Quick Wins & Foundation', 'Core Improvements', 'Advanced Features'],
        'Key Initiatives': [
            '• Callback system implementation\n• Visa rebooking workflow\n• Basic case tracking',
            '• Full case continuity system\n• Language training program\n• IVR optimization',
            '• AI-powered routing\n• Predictive analytics\n• Proactive service'
        ]
    }
    
    roadmap_df = pd.DataFrame(roadmap_data)
    st.dataframe(roadmap_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center">
    <p><strong>Air France Customer Service Intelligence Dashboard</strong></p>
    <p>Data Analysis Project | Demonstrating Business Impact through NLP & Machine Learning</p>
    <p>Analysis Date: {report['analysis_date']}</p>
</div>
""", unsafe_allow_html=True)

