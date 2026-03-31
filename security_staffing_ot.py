import streamlit as st
import pulp
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Security Staffing Optimizer", layout="wide")
st.title("🔧 Federal Security Staffing Optimizer")
st.markdown("Upload Post Exhibit → PuLP optimization with explicit relief + overtime costing")

# Sidebar - Configurable Rules
st.sidebar.header("Labor & Contract Rules")
regular_wage = st.sidebar.number_input("Regular Hourly Wage ($)", min_value=15.0, value=25.0, step=0.5)
ot_multiplier = st.sidebar.number_input("Overtime Multiplier", min_value=1.0, value=1.5, step=0.1)
max_straight_hrs = st.sidebar.number_input("Max Straight-Time Hours/Week", min_value=30, max_value=60, value=40)
relief_percent = st.sidebar.slider("Relief & Break Factor (%)", 10, 40, 20, 5) / 100.0
supervisor_ratio = st.sidebar.number_input("Officers per Supervisor", min_value=4, max_value=20, value=8)
attrition_rate = st.sidebar.slider("Yearly Attrition Rate (%)", 0, 50, 25) / 100.0
desired_buffer = st.sidebar.slider("Desired Headcount Buffer (%)", 0, 30, 10) / 100.0

# File uploader
uploaded_file = st.file_uploader("Upload Post Exhibit (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.subheader("Post Exhibit Preview")
    st.dataframe(df.head(10))
    
    # Define day columns
    day_cols = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_map = {col: i for i, col in enumerate(day_cols)}
    
    # Create Post_ID
    df['Post_ID'] = df['Bldg Address'].astype(str) + " | Post " + df['Post #'].astype(str) + " | " + df['CITY'].astype(str)
    
    # Parse required coverage blocks
    required_data = []
    for _, row in df.iterrows():
        post_id = row['Post_ID']
        start = str(row.get('Start Time', ''))
        end = str(row.get('End Time', ''))
        hrs = float(row.get('Calc Hrs Per Day', 8.0))
        for day_name in day_cols:
            if day_name in row:
                val = str(row[day_name]).strip().lower()
                if val in ['x', '1', '1.0']:
                    required_data.append({
                        'Post_ID': post_id,
                        'Day': day_map[day_name],
                        'Shift_Start': start,
                        'Shift_End': end,
                        'Hrs': hrs,
                        'Required': 1.0
                    })
    
    req_df = pd.DataFrame(required_data)
    st.success(f"Parsed **{len(req_df)}** required shift blocks across **{req_df['Post_ID'].nunique()}** posts.")
    
    if st.button("🚀 Run Optimization (Explicit Relief + OT Costing)"):
        with st.spinner("Optimizing with PuLP..."):
            unique_blocks = req_df[['Post_ID', 'Day', 'Shift_Start', 'Shift_End']].drop_duplicates().reset_index(drop=True)
            
            prob = pulp.LpProblem("Security_Staffing_OT", pulp.LpMinimize)
            
            # Primary officers per unique block
            primary = {i: pulp.LpVariable(f"prim_{i}", lowBound=0, cat='Integer') 
                       for i in range(len(unique_blocks))}
            
            # Relief pool per day + shift start
            relief_slots = req_df[['Day', 'Shift_Start']].drop_duplicates()
            relief = {(d, s): pulp.LpVariable(f"rel_d{d}_s{s}", lowBound=0, cat='Integer') 
                      for d, s in zip(relief_slots['Day'], relief_slots['Shift_Start'])}
            
            total_primary = pulp.lpSum(primary.values())
            total_relief = pulp.lpSum(relief.values())
            
            # Objective: minimize total labor cost (straight + OT premium)
            avg_shift_hrs = req_df['Hrs'].mean() if not req_df.empty else 8.0
            total_hours = (total_primary + total_relief) * avg_shift_hrs
            # Rough but effective cost model: base cost + OT penalty when exceeding straight time
            base_cost = total_hours * regular_wage
            ot_premium = (total_hours - (total_primary * max_straight_hrs)) * regular_wage * (ot_multiplier - 1) * 0.5
            prob += base_cost + max(0, ot_premium)
            
            # Coverage constraints
            for idx, block in unique_blocks.iterrows():
                matching = req_df[(req_df['Post_ID'] == block['Post_ID']) &
                                  (req_df['Day'] == block['Day']) &
                                  (req_df['Shift_Start'] == block['Shift_Start'])]
                req = matching['Required'].sum()
                rel_contrib = relief.get((block['Day'], block['Shift_Start']), 0)
                prob += primary[idx] + rel_contrib >= req * (1 + relief_percent)
            
            # Relief pool constraint
            prob += total_relief >= relief_percent * total_primary * 0.8
            
            # Solve
            status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
            
            if pulp.LpStatus[status] == 'Optimal':
                total_prim_shifts = pulp.value(total_primary)
                total_rel_shifts = pulp.value(total_relief)
                total_hours_week = (total_prim_shifts + total_rel_shifts) * avg_shift_hrs
                
                est_officers = np.ceil(total_hours_week / max_straight_hrs)
                straight_hours = min(total_hours_week, est_officers * max_straight_hrs)
                ot_hours = max(0, total_hours_week - straight_hours)
                
                total_weekly_cost = (straight_hours * regular_wage) + (ot_hours * regular_wage * ot_multiplier)
                
                est_supervisors = np.ceil(est_officers / supervisor_ratio)
                est_relief_officers = np.ceil(total_rel_shifts * avg_shift_hrs / max_straight_hrs)
                total_headcount = int(est_officers + est_supervisors + est_relief_officers) * (1 + desired_buffer)
                
                st.success(f"**Optimal Total Headcount: {int(total_headcount)}** | **Weekly Labor Cost: ${total_weekly_cost:,.2f}**")
                st.write(f"- Base Officers: ~{int(est_officers)}")
                st.write(f"- Supervisors: ~{int(est_supervisors)}")
                st.write(f"- Relief Officers: ~{int(est_relief_officers)}")
                st.write(f"- Estimated Weekly OT Hours: {ot_hours:.1f}")
                
                # Build schedule
                schedule = []
                for i, block in unique_blocks.iterrows():
                    assigned = pulp.value(primary[i])
                    if assigned and assigned > 0:
                        schedule.append({
                            'Type': 'Primary',
                            'Post': block['Post_ID'],
                            'Day': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][block['Day']],
                            'Start': block['Shift_Start'],
                            'End': block['Shift_End'],
                            'Assigned': assigned
                        })
                
                for (d, s), val in relief.items():
                    assigned = pulp.value(val)
                    if assigned and assigned > 0:
                        schedule.append({
                            'Type': 'Relief',
                            'Post': 'Relief Pool',
                            'Day': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d],
                            'Start': s,
                            'End': 'Flexible',
                            'Assigned': assigned
                        })
                
                sched_df = pd.DataFrame(schedule)
                st.subheader("Optimized Schedule (Primary + Relief)")
                st.dataframe(sched_df)
                
                csv = sched_df.to_csv(index=False)
                st.download_button("📥 Download Schedule CSV", csv, "optimized_schedule.csv")
                
                # Recruiting forecast
                st.subheader("12-Month Recruiting Forecast")
                current = total_headcount
                months = list(range(1,13))
                projected = []
                hires = []
                for m in months:
                    lost = current * (attrition_rate / 12)
                    current = max(0, current - lost)
                    projected.append(current)
                    hires.append(lost)
                
                forecast_df = pd.DataFrame({"Month": months, "Projected Headcount": projected, "Hires Needed": hires})
                fig = px.line(forecast_df, x="Month", y=["Projected Headcount", "Hires Needed"], title="Attrition & Recruiting Projection")
                st.plotly_chart(fig)
                st.dataframe(forecast_df.round(1))
            else:
                st.error("Optimization did not converge. Try adjusting relief % or max hours.")
else:
    st.info("Upload your Post Exhibit Excel/CSV to begin.")

st.caption("Built with PuLP for optimal staffing | Explicit relief + OT costing")
