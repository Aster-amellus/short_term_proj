import pandas as pd
import matplotlib.pyplot as plt


# Draw Emissions Trends
def draw_emissions_trends(data, city_name):
    city_data = data[data["city"] == city_name]

    # Convert date to datetime format for better plotting
    city_data["date"] = pd.to_datetime(city_data["date"], format="%d/%m/%Y")

    # Plotting the trends for each sector
    plt.figure(figsize=(14, 8))

    for sector in city_data["sector"].unique():
        sector_data = city_data[city_data["sector"] == sector]
        plt.plot(sector_data["date"], sector_data["value"], label=sector)

    plt.title(f"Emissions Trends for {city_name}")
    plt.xlabel("Date")
    plt.ylabel("Emissions")
    plt.legend(title="Sector")
    plt.grid(True)
    plt.show()


# Draw Emissions Heatmap
def draw_emissions_heatmap(data, city_name):
    city_data = data[data["city"] == city_name]

    # Convert date to datetime format for better plotting
    city_data["date"] = pd.to_datetime(city_data["date"], format="%d/%m/%Y")

    # Pivot the data for heatmap
    heatmap_data = city_data.pivot(index="sector", columns="date", values="value")

    # Plotting the heatmap
    plt.figure(figsize=(14, 8))
    plt.imshow(heatmap_data, cmap="hot", interpolation="nearest")
    plt.title(f"Emissions Heatmap for {city_name}")
    plt.xlabel("Date")
    plt.ylabel("Sector")
    plt.colorbar()
    plt.show()
