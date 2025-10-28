import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from math import radians, sin, cos, sqrt, atan2

# 1. Data Synthesis (Based on Notebook Analysis) 

# Vehicle parameters for multi-objective optimization (Simulated)
VEHICLE_PARAMS = {
    'Truck': {'speed_factor': 0.8, 'cost_rate_per_unit': 1.5, 'co2_rate_per_km': 0.25},
    'Van': {'speed_factor': 1.0, 'cost_rate_per_unit': 1.0, 'co2_rate_per_km': 0.15},
    'Motorcycle': {'speed_factor': 1.2, 'cost_rate_per_unit': 0.5, 'co2_rate_per_km': 0.05},
    'Drone': {'speed_factor': 1.5, 'cost_rate_per_unit': 2.0, 'co2_rate_per_km': 0.01}, # High operational cost, low CO2, high speed
}

def create_mock_data(n_rows=5000):
    np.random.seed(42)
    data = pd.DataFrame({
        # FIX: Ensure Order_ID is a string to prevent TypeError in str.join() later
        'Order_ID': np.arange(1, n_rows + 1).astype(str),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'Order_Status': np.random.choice(['Delivered', 'Delayed', 'Cancelled', 'Pending'], n_rows, p=[0.75, 0.15, 0.05, 0.05]),
        'Delivery_Delay_Hours': np.clip(np.random.normal(loc=15, scale=25, size=n_rows), 0, 150).astype(int),
        'Vehicle_Type': np.random.choice(list(VEHICLE_PARAMS.keys()), n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        'Vehicle_Age_Years': np.random.randint(1, 16, n_rows),
        'Customer_Rating': np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.05, 0.1, 0.2, 0.3, 0.35]),
    })

    # Add mock geographic data for routing simulation (Los Angeles area)
    data['Latitude'] = np.random.uniform(low=33.8, high=34.2, size=n_rows).round(5)
    data['Longitude'] = np.random.uniform(low=-118.5, high=-117.8, size=n_rows).round(5)

    # Simulate CO2 dependency on Vehicle_Age and Type
    data['CO2_per_1000KM'] = (
        data['Vehicle_Age_Years'] * 0.5 +
        data['Vehicle_Type'].map({k: v['co2_rate_per_km'] * 1000 for k, v in VEHICLE_PARAMS.items()}) + # Use base rate * 1000
        np.random.normal(0, 5, n_rows)
    )
    data['CO2_per_1000KM'] = np.clip(data['CO2_per_1000KM'], 5, 50).round(2)

    return data

df = create_mock_data()

#  2. Streamlit Configuration and Layout 

st.set_page_config(
    page_title="Logistics Fleet & Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚛 Smart Route Planner")
st.markdown("Dynamic dashboard for monitoring fleet health, order status, and customer satisfaction.")

# 3. Sidebar Filters 
st.sidebar.header("Global Filters")

# Region Multiselect
all_regions = df['Region'].unique()
selected_regions = st.sidebar.multiselect(
    "Select Region(s)",
    options=all_regions,
    default=all_regions
)

# Vehicle Type Selector (used for CO2 chart)
all_vehicle_types = df['Vehicle_Type'].unique()
selected_vehicle_types = st.sidebar.multiselect(
    "Select Vehicle Type(s) for CO2 Analysis",
    options=all_vehicle_types,
    default=['Truck', 'Van']
)


df_filtered_region = df[df['Region'].isin(selected_regions)]

# Display overall KPIs 
col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)

with col_kpi_1:
    total_orders = df_filtered_region.shape[0]
    st.metric("Total Orders (Selected Regions)", f"{total_orders:,}")

with col_kpi_2:
    avg_rating = df_filtered_region['Customer_Rating'].mean().round(2)
    st.metric("Average Customer Rating", f"{avg_rating} / 5.0")

with col_kpi_3:
    on_time_rate = (df_filtered_region['Order_Status'] == 'Delivered').mean() * 100
    st.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")

st.markdown("---")


#  4. Dashboard Visualizations 

# Row 1: Chart 1 and Chart 2
chart_col1, chart_col2 = st.columns(2)


# CHART 1: Bar Chart - Orders by Status (Filter: Region)

with chart_col1:
    st.subheader("1. Order Status Distribution by Region")

    
    status_counts = df_filtered_region.groupby(['Order_Status', 'Region']).size().reset_index(name='Count')

    base = alt.Chart(status_counts).encode(
        x=alt.X('Count', title='Number of Orders'),
        y=alt.Y('Order_Status', title='Order Status', sort='-x'),
        tooltip=['Order_Status', 'Region', 'Count']
    ).properties(height=300)

    bars = base.mark_bar().encode(
        color=alt.Color('Region', title='Region'),
        order=alt.Order('Region', sort='ascending')
    ).interactive()

    st.altair_chart(bars, use_container_width=True)


# CHART 2: Donut Chart - Customer Rating Breakdown (Dynamic: Region Selection)

with chart_col2:
    st.subheader("2. Customer Satisfaction Rating Breakdown (Dynamic: Region Selection)")

    
    rating_counts = df_filtered_region.groupby('Customer_Rating').size().reset_index(name='Count')
    rating_counts['Customer_Rating'] = rating_counts['Customer_Rating'].astype(str) # Convert to string for discrete colors

    base = alt.Chart(rating_counts).encode(
        theta=alt.Theta("Count", stack=True)
    ).properties(height=300)

    pie = base.mark_arc(outerRadius=120, innerRadius=80).encode( # Inner radius makes it a donut
        color=alt.Color("Customer_Rating", title="Rating", scale=alt.Scale(domain=['5', '4', '3', '2', '1'], range=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c'])),
        order=alt.Order("Customer_Rating", sort="descending"),
        tooltip=["Customer_Rating", "Count", alt.Tooltip("Count", format=".1%")]
    )

    text = base.mark_text(radius=140).encode(
        text=alt.Text("Count"),
        order=alt.Order("Count", sort="descending"),
        color=alt.value("black")
    )

    st.altair_chart(pie + text, use_container_width=True)

st.markdown("---")

# Row 2: Chart 3 and Chart 4
chart_col3, chart_col4 = st.columns(2)


# CHART 3: Scatter Plot - Fleet Health (Filter: Vehicle Type)

with chart_col3:
    st.subheader("3. Vehicle Age vs. CO2 Emissions (Interactive Filter)")

    df_filtered_vehicle = df[df['Vehicle_Type'].isin(selected_vehicle_types)]

    scatter = alt.Chart(df_filtered_vehicle).mark_circle(size=60).encode(
        x=alt.X('Vehicle_Age_Years', title='Vehicle Age (Years)'),
        y=alt.Y('CO2_per_1000KM', title='CO2 Emissions (Kg/1000KM)'),
        color=alt.Color('Vehicle_Type', title='Vehicle Type'),
        tooltip=['Vehicle_Type', 'Vehicle_Age_Years', 'CO2_per_1000KM']
    ).properties(height=350).interactive() # Enable zooming and panning

    st.altair_chart(scatter, use_container_width=True)



# CHART 4: Histogram - Delivery Delay Distribution 

with chart_col4:
    st.subheader("4. Distribution of Delivery Delays (Hours)")

    # Slider for delay threshold
    delay_threshold = st.slider(
        "Highlight Deliveries Delayed More Than (Hours)",
        min_value=0,
        max_value=100,
        value=24,
        step=1,
        key='delay_slider_chart4' # Added unique key since we are now in chart_col4
    )

    
    df_temp = df_filtered_region.copy()
    df_temp['Delay_Highlight'] = np.where(
        df_temp['Delivery_Delay_Hours'] > delay_threshold,
        f'> {delay_threshold} Hours',
        f'<= {delay_threshold} Hours'
    )

    
    hist = alt.Chart(df_temp).mark_bar().encode(
        x=alt.X('Delivery_Delay_Hours', bin=alt.Bin(maxbins=50), title="Delivery Delay (Hours)"),
        y=alt.Y('count()', title="Number of Orders"),
        tooltip=['Delivery_Delay_Hours', 'count()'],
        color=alt.Color('Delay_Highlight',
            scale=alt.Scale(domain=[f'> {delay_threshold} Hours', f'<= {delay_threshold} Hours'],
                            range=['#e74c3c', '#3498db']),
            title="Delay Status"
        )
    ).properties(height=350)

    st.altair_chart(hist, use_container_width=True)


# 5. Intelligent Routing System (Simulated) 

st.markdown("---")
st.header("5.  Multi-Objective Intelligent Route Planning") # Renumbered to 5
st.markdown("Optimize the delivery route by setting weights for **Financial Cost**, **Time/Distance**, and **Environmental Impact (CO2)**.")


# Function to calculate Euclidean distance (proxy for physical distance)
def calculate_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

#  to calculate a single weighted cost
def calculate_weighted_cost(distance, vehicle_type, W_cost, W_time, W_co2):
    params = VEHICLE_PARAMS[vehicle_type]

    time_cost = distance * (1 / params['speed_factor'])
    
    co2_cost = distance * params['co2_rate_per_km']

    financial_cost = distance * params['cost_rate_per_unit']

    NORM_DIST = 1.0
    NORM_CO2 = 0.3
    NORM_FINANCIAL = 2.0

    normalized_time_cost = time_cost / NORM_DIST
    normalized_co2_cost = co2_cost / NORM_CO2
    normalized_financial_cost = financial_cost / NORM_FINANCIAL

    total_weighted_cost = (
        (W_time / 100) * normalized_time_cost +
        (W_co2 / 100) * normalized_co2_cost +
        (W_cost / 100) * normalized_financial_cost
    )
    return total_weighted_cost, financial_cost, time_cost, co2_cost


# Multi-Objective Nearest Neighbor Heuristic
def find_optimized_route(df_stops, start_lat, start_lon, vehicle_type, W_cost, W_time, W_co2):
    # Add depot as the starting point
    depot = pd.DataFrame([{'Latitude': start_lat, 'Longitude': start_lon, 'Order_ID': 'Depot'}])
    stops = pd.concat([depot, df_stops[['Order_ID', 'Latitude', 'Longitude']]]).reset_index(drop=True)

    num_stops = len(stops)
    if num_stops <= 1:
        stops['Sequence'] = stops.index
        return stops, 0, 0, 0 # Return stops, total_cost, total_time, total_co2

    unvisited = stops.index.tolist()
    current_index = 0  # Start at Depot (index 0)
    route_indices = [current_index]
    unvisited.remove(current_index)

    total_optimized_cost = 0
    total_financial_cost = 0
    total_time_cost = 0
    total_co2_cost = 0

    while unvisited:
        current_stop = stops.iloc[current_index]
        min_weighted_cost = float('inf')
        nearest_index = -1
        best_segment_metrics = (0, 0, 0) # financial, time, co2

        for next_index in unvisited:
            next_stop = stops.iloc[next_index]
            distance = calculate_distance(current_stop['Latitude'], current_stop['Longitude'],
                                          next_stop['Latitude'], next_stop['Longitude'])

            weighted_cost, financial_cost, time_cost, co2_cost = calculate_weighted_cost(
                distance, vehicle_type, W_cost, W_time, W_co2
            )

            if weighted_cost < min_weighted_cost:
                min_weighted_cost = weighted_cost
                nearest_index = next_index
                best_segment_metrics = (financial_cost, time_cost, co2_cost)

        # Update totals for the selected segment
        total_financial_cost += best_segment_metrics[0]
        total_time_cost += best_segment_metrics[1]
        total_co2_cost += best_segment_metrics[2]

        current_index = nearest_index
        route_indices.append(current_index)
        unvisited.remove(current_index)

    # Return to depot (closing the loop)
    
    last_stop = stops.iloc[route_indices[-1]]
    depot_stop = stops.iloc[0]
    distance_to_depot = calculate_distance(last_stop['Latitude'], last_stop['Longitude'],
                                           depot_stop['Latitude'], depot_stop['Longitude'])

    _, final_financial_cost, final_time_cost, final_co2_cost = calculate_weighted_cost(
        distance_to_depot, vehicle_type, W_cost, W_time, W_co2
    )

    total_financial_cost += final_financial_cost
    total_time_cost += final_time_cost
    total_co2_cost += final_co2_cost
    route_indices.append(0) # Index 0 is the Depot

    
    optimized_route_df = stops.iloc[route_indices].copy()
    optimized_route_df.reset_index(drop=True, inplace=True)
    
    optimized_route_df['Sequence'] = optimized_route_df.index

    return optimized_route_df, total_financial_cost, total_time_cost, total_co2_cost


# Routing UI and Logic 

route_col1, route_col2, route_col3 = st.columns(3)

# Simulated Depot Location
depot_location = {
    'Depot A (LA)': {'lat': 34.05, 'lon': -118.25},
    'Depot B (Riverside)': {'lat': 33.95, 'lon': -117.38},
}
depot_names = list(depot_location.keys())

with route_col1:
    selected_depot_name = st.selectbox("1. Select Starting Depot", depot_names)
    start_lat = depot_location[selected_depot_name]['lat']
    start_lon = depot_location[selected_depot_name]['lon']

with route_col2:
    selected_vehicle_type = st.selectbox("2. Select Vehicle for Route", list(VEHICLE_PARAMS.keys()))
    num_stops = st.slider("3. Number of Deliveries to Optimize", min_value=3, max_value=25, value=10, step=1)

with route_col3:
    st.markdown("4. **Set Optimization Weights (Total must be 100)**")
    # Using columns for sliders to save space
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        W_cost = st.slider("Cost (%)", 0, 100, 33, key='w_cost')
    with w_col2:
        W_time = st.slider("Time (%)", 0, 100, 34, key='w_time')
    with w_col3:
        W_co2 = st.slider("CO2 (%)", 0, 100, 33, key='w_co2')

    
    total_weights = W_cost + W_time + W_co2
    if total_weights != 100:
        st.warning(f"Total weights: {total_weights}%. Adjust sliders to reach 100%.")


# Get the first N orders 
df_stops_to_optimize = df_filtered_region.head(num_stops)

# Calculate the route
optimized_route_df, total_financial_cost, total_time_cost, total_co2_cost = find_optimized_route(
    df_stops_to_optimize, start_lat, start_lon, selected_vehicle_type, W_cost, W_time, W_co2
)

st.markdown("---")

# Display the new multi-objective metrics
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Total Estimated Cost (Units)", f"{total_financial_cost:.2f}")

with metric_col2:
    st.metric("Total Estimated Time (Units)", f"{total_time_cost:.2f}")

with metric_col3:
    st.metric("Total Estimated CO2 (Kg)", f"{total_co2_cost:.2f}")


# Visualization
st.subheader("Optimized Delivery Route Map")

# Base map layer showing all stops + depot
base_map = alt.Chart(optimized_route_df).encode(
    latitude='Latitude',
    longitude='Longitude',
    tooltip=['Order_ID']
)

# 1. Route Path

route_line = base_map.mark_line(color='#e74c3c').encode(
    order=alt.Order('Sequence'), 
    size=alt.value(3)
)

# 2. Stop Points
stop_points = base_map.mark_circle(size=150, opacity=0.8).encode(
    color=alt.Color('Order_ID', scale=alt.Scale(range=['#3498db']), legend=None),
    size=alt.value(150),
    tooltip=['Order_ID']
).transform_filter(
    
    alt.FieldEqualPredicate(field='Order_ID', equal='Depot')
)

# 3. Depot Marker (Start/End)
depot_marker = base_map.mark_square(size=200, color='#f1c40f', stroke='black', strokeWidth=2).encode(
    tooltip=alt.Tooltip('Order_ID', title='Type')
).transform_filter(
    alt.FieldEqualPredicate(field='Order_ID', equal='Depot')
)


final_route_chart = (route_line + stop_points + depot_marker).properties(
    title=f"Optimized Route from {selected_depot_name} using {selected_vehicle_type}"
).interactive() # Allows panning/zooming

st.altair_chart(final_route_chart, use_container_width=True)

st.markdown("**Optimized Delivery Sequence:**")
# The last stop is a return to the Depot, so exclude it from the delivery list
route_sequence = " → ".join(optimized_route_df['Order_ID'].tolist()[:-1])
st.info(route_sequence)

st.markdown("---")
st.caption("Data is synthetic and designed to simulate logistics performance metrics.")
