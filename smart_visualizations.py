"""
📊 Advanced Data Visualization Engine
======================================
Beautiful, interactive, and intelligent visualizations for financial data

Features:
- 3D interactive charts
- Animated transitions
- Real-time updates
- Smart color schemes
- Responsive design
- Advanced statistical plots
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SmartVisualizer:
    """Ultra-smart visualization engine with beautiful charts"""
    
    def __init__(self):
        # Premium color palettes
        self.color_schemes = {
            'gradient': ['#667eea', '#764ba2', '#f093fb', '#4facfe'],
            'professional': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            'modern': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
            'elegant': ['#2C3E50', '#8E44AD', '#3498DB', '#1ABC9C']
        }
        
        self.default_layout = dict(
            font=dict(family="Poppins, sans-serif", size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode='closest'
        )
    
    def create_3d_spending_surface(self, df, date_col, category_col, amount_col):
        """Create stunning 3D surface plot of spending patterns"""
        
        # Prepare data for 3D visualization
        df['date'] = pd.to_datetime(df[date_col])
        df['day'] = df['date'].dt.day
        df['category_num'] = pd.Categorical(df[category_col]).codes
        
        # Create pivot for surface
        pivot = df.pivot_table(
            values=amount_col,
            index='day',
            columns='category_num',
            aggfunc='sum',
            fill_value=0
        )
        
        fig = go.Figure(data=[go.Surface(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            contours={
                "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
            }
        )])
        
        fig.update_layout(
            title='3D Spending Landscape',
            scene=dict(
                xaxis_title='Categories',
                yaxis_title='Day of Month',
                zaxis_title='Amount ($)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            **self.default_layout,
            height=600
        )
        
        return fig
    
    def create_animated_category_race(self, df, date_col, category_col, amount_col):
        """Create animated bar chart race of category spending over time"""
        
        df['date'] = pd.to_datetime(df[date_col])
        df['week'] = df['date'].dt.isocalendar().week
        
        # Aggregate by week and category
        weekly = df.groupby(['week', category_col])[amount_col].sum().reset_index()
        weekly = weekly.sort_values(['week', amount_col], ascending=[True, False])
        
        fig = px.bar(
            weekly,
            x=amount_col,
            y=category_col,
            animation_frame='week',
            orientation='h',
            color=category_col,
            color_discrete_sequence=self.color_schemes['gradient'],
            title='Weekly Category Spending Race'
        )
        
        fig.update_layout(
            **self.default_layout,
            height=500,
            showlegend=False,
            xaxis_title='Amount ($)',
            yaxis_title='Category'
        )
        
        return fig
    
    def create_smart_treemap(self, category_breakdown):
        """Create intelligent hierarchical treemap"""
        
        categories = list(category_breakdown.keys())
        amounts = list(category_breakdown.values())
        
        # Create hierarchy
        df = pd.DataFrame({
            'Category': categories,
            'Amount': amounts,
            'Parent': ['Total'] * len(categories)
        })
        
        # Add total
        total_row = pd.DataFrame({
            'Category': ['Total'],
            'Amount': [sum(amounts)],
            'Parent': ['']
        })
        
        df = pd.concat([df, total_row], ignore_index=True)
        
        fig = px.treemap(
            df,
            names='Category',
            parents='Parent',
            values='Amount',
            color='Amount',
            color_continuous_scale='RdYlGn_r',
            title='Spending Hierarchy (Click to explore)'
        )
        
        fig.update_layout(**self.default_layout, height=500)
        fig.update_traces(
            textinfo="label+value+percent parent",
            textfont=dict(size=14, family="Poppins"),
            marker=dict(line=dict(width=2, color='white'))
        )
        
        return fig
    
    def create_sankey_flow(self, df, category_col, amount_col):
        """Create beautiful Sankey diagram showing money flow"""
        
        # Categorize spending levels
        df['spending_level'] = pd.cut(
            df[amount_col],
            bins=[0, 50, 100, 200, float('inf')],
            labels=['Low ($0-50)', 'Medium ($50-100)', 'High ($100-200)', 'Very High ($200+)']
        )
        
        # Prepare Sankey data
        category_to_idx = {cat: idx for idx, cat in enumerate(df[category_col].unique())}
        level_to_idx = {level: idx + len(category_to_idx) for idx, level in enumerate(df['spending_level'].cat.categories)}
        
        source = [category_to_idx[cat] for cat in df[category_col]]
        target = [level_to_idx[level] for level in df['spending_level']]
        value = df[amount_col].tolist()
        
        labels = list(category_to_idx.keys()) + list(level_to_idx.keys())
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=self.color_schemes['modern']
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color='rgba(100,149,237,0.4)'
            )
        )])
        
        fig.update_layout(
            title='Money Flow Analysis',
            **self.default_layout,
            height=600
        )
        
        return fig
    
    def create_heatmap_calendar(self, df, date_col, amount_col):
        """Create calendar heatmap showing daily spending intensity"""
        
        df['date'] = pd.to_datetime(df[date_col])
        daily_spending = df.groupby('date')[amount_col].sum().reset_index()
        
        # Create calendar grid
        daily_spending['day_of_week'] = daily_spending['date'].dt.dayofweek
        daily_spending['week_of_year'] = daily_spending['date'].dt.isocalendar().week
        
        pivot = daily_spending.pivot(
            index='day_of_week',
            columns='week_of_year',
            values=amount_col
        ).fillna(0)
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=days,
            colorscale='Reds',
            hoverongaps=False,
            hovertemplate='Week %{x}<br>%{y}<br>Spent: $%{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Daily Spending Calendar',
            xaxis_title='Week of Year',
            yaxis_title='Day of Week',
            **self.default_layout,
            height=400
        )
        
        return fig
    
    def create_radar_chart(self, category_breakdown):
        """Create beautiful radar chart for category comparison"""
        
        categories = list(category_breakdown.keys())
        values = list(category_breakdown.values())
        
        # Normalize to 0-100 scale
        max_val = max(values)
        normalized = [v / max_val * 100 for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized,
            theta=categories,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8, color='#667eea'),
            name='Spending'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='lightgray'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False,
            title='Category Spending Radar',
            **self.default_layout,
            height=500
        )
        
        return fig
    
    def create_waterfall_chart(self, category_breakdown):
        """Create waterfall chart showing spending build-up"""
        
        categories = list(category_breakdown.keys())
        values = list(category_breakdown.values())
        
        # Sort by value
        sorted_data = sorted(zip(categories, values), key=lambda x: x[1], reverse=True)
        categories, values = zip(*sorted_data)
        
        # Create measure types
        measure = ['relative'] * len(categories)
        measure.append('total')
        
        fig = go.Figure(go.Waterfall(
            name="Spending",
            orientation="v",
            measure=measure,
            x=list(categories) + ['Total'],
            textposition="outside",
            text=[f"${v:,.0f}" for v in values] + [f"${sum(values):,.0f}"],
            y=list(values) + [sum(values)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#FF6B6B"}},
            totals={"marker": {"color": "#667eea"}}
        ))
        
        fig.update_layout(
            title="Spending Waterfall",
            showlegend=False,
            **self.default_layout,
            height=500,
            xaxis_title="Category",
            yaxis_title="Amount ($)"
        )
        
        return fig
    
    def create_violin_plot(self, df, category_col, amount_col):
        """Create violin plot showing distribution by category"""
        
        fig = go.Figure()
        
        categories = df[category_col].unique()
        colors = px.colors.qualitative.Set3
        
        for i, category in enumerate(categories):
            category_data = df[df[category_col] == category][amount_col]
            
            fig.add_trace(go.Violin(
                y=category_data,
                name=category,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.6,
                x0=category
            ))
        
        fig.update_layout(
            title='Spending Distribution by Category',
            yaxis_title='Amount ($)',
            xaxis_title='Category',
            **self.default_layout,
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_funnel_chart(self, category_breakdown):
        """Create funnel chart for spending priority"""
        
        # Sort categories by amount
        sorted_items = sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True)
        categories, values = zip(*sorted_items)
        
        fig = go.Figure(go.Funnel(
            y=list(categories),
            x=list(values),
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(
                color=self.color_schemes['gradient'],
                line=dict(width=2, color='white')
            ),
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
        ))
        
        fig.update_layout(
            title='Spending Priority Funnel',
            **self.default_layout,
            height=500
        )
        
        return fig
