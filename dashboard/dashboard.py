
# --- Library Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency


# --- Visualization Style & Constants ---
sns.set(style='white', font_scale=1.2, color_codes=True)

DATA_PATH = "dashboard/all_data.csv"
COMPANY_LOGO = "https://cdn.brandfetch.io/idvSn4Org5/w/1200/h/1200/theme/dark/icon.jpeg?c=1bxid64Mup7aczewSAYMX&t=1740864715366"
PRIMARY_COLOR = "#90CAF9"
SECONDARY_COLORS = ["#D3D3D3"] * 9
ALL_COLORS = [PRIMARY_COLOR] + SECONDARY_COLORS


def create_daily_orders_df(df):
    """Create a dataframe with daily order counts and revenue."""
    daily = (
        df.resample('D', on='order_purchase_timestamp')
        .agg(order_count=('order_id', 'nunique'), revenue=('price', 'sum'))
        .reset_index()
    )
    return daily


def create_count_order_items_df(df, group_by_column):
    """Create a dataframe with total quantities sold by group (e.g., category/city)."""
    return (
        df.groupby(group_by_column, as_index=False)["order_id"].count()
        .rename(columns={"order_id": "order_count"})
        .sort_values("order_count", ascending=False)
    )

def create_sum_order_items_df(df, group_by_column):
    """Create a dataframe with total revenue by group (e.g., category/city)."""
    return (
        df.groupby(group_by_column, as_index=False)["price"].sum()
        .rename(columns={"price": "revenue"})
        .sort_values("revenue", ascending=False)
    )


def load_and_preprocess_data():
    """Load and preprocess the data from CSV file."""
    df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"])
    df.sort_values(by="order_purchase_timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def filter_data(df, start_date, end_date):
    """Filter data based on the selected date range."""
    mask = (df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) & \
           (df["order_purchase_timestamp"] <= pd.to_datetime(end_date))
    return df.loc[mask]

# Visualization functions

def plot_daily_orders(df):
    """Plot daily order counts."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["order_purchase_timestamp"], df["order_count"], marker='o', linewidth=2, color=PRIMARY_COLOR)
    ax.set(xlabel="Date", ylabel="Order Count", title="Daily Orders")
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=10)
    return fig


def plot_product_categories_sales_performance(df):
    """Plot best and worst performing product categories by sales volume."""
    fig, ax = plt.subplots(1, 2, figsize=(28, 10))
    # Best
    sns.barplot(x="order_count", y="product_category_name_english", data=df.head(10), palette=ALL_COLORS, ax=ax[0])
    ax[0].set(xlabel="Number of Sales", ylabel=None, title="Top 10 Product Categories (Sales)")
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].tick_params(axis='x', labelsize=14)
    # Worst
    sns.barplot(x="order_count", y="product_category_name_english", data=df.tail(10).sort_values("order_count"), palette=ALL_COLORS, ax=ax[1])
    ax[1].set(xlabel="Number of Sales", ylabel=None, title="Bottom 10 Product Categories (Sales)")
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].tick_params(axis='x', labelsize=14)
    return fig


def plot_product_categories_revenue_performance(df):
    """Plot best and worst performing product categories by revenue."""
    fig, ax = plt.subplots(1, 2, figsize=(28, 10))
    # Best
    sns.barplot(x="revenue", y="product_category_name_english", data=df.head(10), palette=ALL_COLORS, ax=ax[0])
    ax[0].set(xlabel="Total Revenue (R$)", ylabel=None, title="Top 10 Product Categories (Revenue)")
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].tick_params(axis='x', labelsize=14)
    # Worst
    sns.barplot(x="revenue", y="product_category_name_english", data=df.tail(10).sort_values("revenue"), palette=ALL_COLORS, ax=ax[1])
    ax[1].set(xlabel="Total Revenue (R$)", ylabel=None, title="Bottom 10 Product Categories (Revenue)")
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].tick_params(axis='x', labelsize=14)
    return fig


def plot_seller_city_sales_performance(df, value_col, title_prefix):
    """Plot best and worst performing seller cities by sales or revenue."""
    fig, ax = plt.subplots(1, 2, figsize=(28, 10))
    # Best
    sns.barplot(x=value_col, y="seller_city", data=df.head(10), palette=ALL_COLORS, ax=ax[0])
    ax[0].set(xlabel=title_prefix, ylabel=None, title=f"Top 10 Seller Cities ({title_prefix})")
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].tick_params(axis='x', labelsize=14)
    # Worst
    sns.barplot(x=value_col, y="seller_city", data=df.tail(10).sort_values(value_col), palette=ALL_COLORS, ax=ax[1])
    ax[1].set(xlabel=title_prefix, ylabel=None, title=f"Bottom 10 Seller Cities ({title_prefix})")
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].tick_params(axis='x', labelsize=14)
    return fig


def create_sidebar(df):
    """Create and configure the sidebar."""
    with st.sidebar:
        st.image(COMPANY_LOGO)
        min_date = df["order_purchase_timestamp"].min().date()
        max_date = df["order_purchase_timestamp"].max().date()
        start_date, end_date = st.date_input(
            label='Date Range',
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
    return start_date, end_date


def display_metrics(daily_orders_df):
    """Display key metrics in the dashboard."""
    total_orders = int(daily_orders_df.order_count.sum())
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "BRL", locale='pt_BR')
    col1, col2 = st.columns(2)
    col1.metric("Total Orders", value=total_orders)
    col2.metric("Total Revenue", value=total_revenue)


def main():
    """Main function to run the Streamlit dashboard."""
    # Load and preprocess data
    all_df = load_and_preprocess_data()

    # Sidebar: date filter
    start_date, end_date = create_sidebar(all_df)
    main_df = filter_data(all_df, start_date, end_date)


    # --- Custom Header ---
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 0.5em;'>
            <img src='https://cdn.brandfetch.io/idvSn4Org5/w/1200/h/1200/theme/dark/icon.jpeg?c=1bxid64Mup7aczewSAYMX&t=1740864715366' width='80' style='border-radius: 20px; margin-bottom: 0.5em;'>
            <h1 style='color: #1976D2; margin-bottom: 0;'>Olist E-Commerce Dashboard 🚀</h1>
            <p style='color: #555; font-size: 1.2em;'>Analisis penjualan, kategori produk, dan performa kota penjual di marketplace Olist.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Prepare Dataframes ---
    daily_orders_df = create_daily_orders_df(main_df)
    count_order_items_df = create_count_order_items_df(main_df, "product_category_name_english")
    sum_order_items_df = create_sum_order_items_df(main_df, "product_category_name_english")
    count_order_items_by_seller_city_df = create_count_order_items_df(main_df, "seller_city")
    sum_order_items_by_seller_city_df = create_sum_order_items_df(main_df, "seller_city")

    # --- Metrics & Daily Orders ---
    st.markdown("### Ringkasan Penjualan Harian")
    display_metrics(daily_orders_df)
    st.pyplot(plot_daily_orders(daily_orders_df))

    # --- Product Categories Performance ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top & Bottom Product Categories (By Sales)")
        st.pyplot(plot_product_categories_sales_performance(count_order_items_df))
    with col2:
        st.markdown("#### Top & Bottom Product Categories (By Revenue)")
        st.pyplot(plot_product_categories_revenue_performance(sum_order_items_df))

    # --- Seller Cities Performance ---
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Top & Bottom Seller Cities (By Sales)")
        st.pyplot(plot_seller_city_sales_performance(count_order_items_by_seller_city_df, "order_count", "Number of Sales"))
    with col4:
        st.markdown("#### Top & Bottom Seller Cities (By Revenue)")
        st.pyplot(plot_seller_city_sales_performance(sum_order_items_by_seller_city_df, "revenue", "Total Revenue (R$)"))

    # --- Insight Box ---
    st.info(
        """
        **Insight:**
        - Kategori produk dan kota penjual dengan performa terbaik didominasi oleh kategori dan kota besar.
        - São Paulo mendominasi baik dari sisi pendapatan maupun volume penjualan.
        - Kategori Health & Beauty dan Bed Bath Table selalu masuk 3 besar baik dari sisi revenue maupun sales volume.
        """
    )

    # --- Footer ---
    st.markdown(
        """
        <hr style='margin-top:2em;margin-bottom:0.5em;'>
        <div style='text-align:center; color:#888;'>
            Copyright © Olist 2021 &nbsp;|&nbsp; <a href='mailto:furqoncreative24@gmail.com'>Contact</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()