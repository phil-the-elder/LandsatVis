import os
import tarfile
import pandas as pd
import tifffile as tiff
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer


def get_rgb(directory, ID, contrast=0.3):
    """
    This function loads and processes the Red, Green, and Blue (RGB) bands from a Landsat scene.
    It handles both Landsat 5, 7, 8, and 9 satellites.

    Parameters:
    directory (str): The path to the directory containing the Landsat scene data.
    ID (str): The Landsat scene ID.
    contrast (float, optional): The contrast level for the RGB bands. Default is 0.3.

    Returns:
    numpy.ndarray: A 3D numpy array representing the RGB image.
    """

    # Load Blue (B2), Green (B3) and Red (B4) bands
    if ID[0:4] in ['LT05', 'LE07']:
        # Landsat 5 and 7
        R = tiff.imread(os.path.join(directory, 'data', f'{ID}_SR_B3.TIF'))
        G = tiff.imread(os.path.join(directory, 'data', f'{ID}_SR_B2.TIF'))
        B = tiff.imread(os.path.join(directory, 'data', f'{ID}_SR_B1.TIF'))
    else:
        # Landsat 8 and 9
        R = tiff.imread(os.path.join(directory, 'data', f'{ID}_SR_B4.TIF'))
        G = tiff.imread(os.path.join(directory, 'data', f'{ID}_SR_B3.TIF'))
        B = tiff.imread(os.path.join(directory, 'data', f'{ID}_SR_B2.TIF'))

    # Stack and scale bands
    RGB = np.dstack((R, G, B))
    RGB = np.clip(RGB*0.0000275-0.2, 0, 1)

    # Clip to enhance contrast
    RGB = np.clip(RGB, 0, contrast) / contrast

    return RGB


def get_datasets(api, lat, long, datasets=["landsat_tm_c2_l2", "landsat_etm_c2_l2", "landsat_ot_c2_l2"],
                 start_date='1980-01-01', end_date='2024-12-31'):
    """
    This function retrieves Landsat scenes based on the given parameters.

    Parameters:
    api (landsatxplore.api.API): An instance of the Landsat API.
    lat (float): The latitude of the area of interest.
    long (float): The longitude of the area of interest.
    datasets (list, optional): A list of Landsat datasets to search.
        Default is ["landsat_tm_c2_l2", "landsat_etm_c2_l2", "landsat_ot_c2_l2"].
    start_date (str, optional): The start date for the search. Default is '1980-01-01'.
    end_date (str, optional): The end date for the search. Default is '2024-12-31'.

    Returns:
    list: A list of dictionaries, where each dictionary represents a Landsat scene.
    """
    results = []
    # Datasets for Landsat 5, 7, 8 and 9

    for dataset in datasets:
        scenes = api.search(
            dataset=dataset,
            latitude=lat,
            longitude=long,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=100,
            max_results=10000
        )

        results.extend(scenes)
    return results


def format_data(df, season, hemisphere, quality_level='standard'):
    """
    This function filters and processes a DataFrame containing Landsat scene data based on the given parameters.
    It selects scenes based on season, hemisphere, quality level, and other conditions.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing Landsat scene data. It should have columns: 'display_id', 'wrs_path',
        'wrs_row', 'satellite', 'cloud_cover', 'tier', and 'acquisition_date'.
    season (str): The season to filter scenes for. It can be one of the following: 'winter', 'spring', 'summer', 'fall'.
    hemisphere (str): The hemisphere to filter scenes for. It can be either 'northern' or 'southern'.
    quality_level (str, optional): The quality level to filter scenes for.
        It can be one of the following: 'high', 'standard', 'low'. Default is 'standard'.

    Returns:
    pandas.DataFrame: A filtered and processed DataFrame containing the selected scenes.
    It includes columns: 'id', 'path', 'row', 'satellite', 'cloud_cover', 'tier', 'date', 'year', 'month'.
    """
    # Get scene tier
    df['tier'] = [int(x[-1]) for x in df["display_id"]]
    # Get satellite
    df['satellite'] = [int(str(x)[-1]) for x in df['satellite']]
    # Get necessary columns
    df = df[['display_id', 'wrs_path', 'wrs_row', 'satellite', 'cloud_cover', 'tier', 'acquisition_date']]
    df.columns = ['id', 'path', 'row', 'satellite', 'cloud_cover', 'tier', 'date']
    # Get year and month
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df.sort_values('date', ascending=True, inplace=True)

    # only tile (175, 83)
    # df = df[df.row == tile_number]
    season_list = ['winter', 'spring', 'summer', 'fall']
    if hemisphere == 'southern':
        season_list = ['summer', 'fall', 'winter', 'spring']
    season_index = season_list.index(season.lower()) + 1
    df = df[df.month % 12 // 3 + 1 == season_index]

    quality_levels = ['high', 'standard', 'low']
    df = df[df.tier <= quality_levels.index(quality_level.lower()) + 1]

    # before 31 May 2003 for landsat 7 due to lens error after that date
    df = df[(df.satellite != 7) | (df.date < '2003-05-31')]
    # no landsat 4
    df = df[df.satellite != 4]
    # sort by cloud cover and month
    df = df.sort_values(['year', 'cloud_cover'], ascending=True)
    # get first entry for each year
    df = df.groupby('year').first().reset_index()

    return df


def download_scene(EarthExplorer, ID, directory, utmx, utmy, height=500, width=500):
    """
    Downloads and processes a Landsat scene to obtain a cropped RGB image of a specified area of interest.

    Parameters:
    EarthExplorer: An instance of the EarthExplorer class for downloading Landsat data.
    ID (str): The Landsat scene ID.
    directory (str): The path to the directory where the Landsat data will be stored.
    utmx (float): The UTM x-coordinate of the area of interest.
    utmy (float): The UTM y-coordinate of the area of interest.
    height (int, optional): The height of the cropped RGB image. Default is 500.
    width (int, optional): The width of the cropped RGB image. Default is 500.

    Returns:
    tuple: A tuple containing the cropped RGB image, UTM x-coordinate, and UTM y-coordinate of the area of interest.
    """
    print(f"ID: {ID}")
    crop_rgb = None
    image = os.path.join(directory, 'data', f'{ID}.tar')
    # Download the scene
    try:
        if not os.path.isfile(image):
            EarthExplorer.download(ID, output_dir=os.path.join(directory, 'data'))
            print('{} succesful'.format(ID))
    # Additional error handling
    except Exception as e:
        if os.path.isfile(image):
            print('{} error but file exists'.format(ID))
        else:
            print('{} error'.format(ID))
            print(str(e))
    if os.path.isfile(image):
        # Extract files from tar archive
        try:
            tar = tarfile.open(image)
            tar.extractall(path=os.path.join(directory, 'data'))
            tar.close()
        except Exception as e:
            print('Error extracting tar archive:', str(e))
    if os.path.isfile(os.path.join(directory, 'data', f'{ID}_SR_B3.TIF')):
        x, y = 2500, 5500
        # Get RGB image
        rgb = get_rgb(directory, ID)

        # Get pixel coordinates of area of interest
        band = rio.open(os.path.join(directory, 'data', f'{ID}_SR_B3.TIF'))
        if utmx is None and utmy is None:
            utmx, utmy = band.xy(y, x)
        y, x = band.index(utmx, utmy)

        # Crop area of interest
        crop_rgb = rgb[y:y + height, x:x + width]

    return crop_rgb, utmx, utmy


def main():
    """
    This function is the main entry point for the Landsat time-series visualization script.
    It initializes the Landsat API, Earth Explorer, retrieves Landsat scenes, processes the data,
    downloads and processes each scene, and creates an animated GIF of the time-series.

    Parameters:
    None

    Returns:
    None
    """
    username = os.environ.get('USGS_USERNAME')
    password = os.environ.get('USGS_PASSWORD')
    app_folder = r'C:\Projects\_Training\20240815_DataVis\SatelliteGifs\LandsatVis'
    output_name = 'landsat_timeseries_denver.gif'
    latitude = 39.77983697215904
    longitude = -105.00939118976989

    # Initialize a new API instance
    api = API(username, password)

    # Initialize the Earth Explorer instance with your USGS credentials
    ee = EarthExplorer(username, password)

    datasets = get_datasets(api, latitude, longitude)

    # Create a DataFrame from the scenes
    df_scenes = format_data(pd.DataFrame(datasets), 'summer', 'northern', 'high')

    images = []
    utmx = None
    utmy = None

    try:
        for ID in df_scenes.id:
            img, utmx, utmy = download_scene(ee, ID, app_folder, utmx, utmy)
            if img is not None:
                images.append(img)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(xlim=(0, 500), ylim=(0, 500))

        def animate(i, images, df_scenes):
            """Returns the i-th frame of the animation"""
            ax.imshow(images[i])
            title = f'Year: {str(df_scenes.year[i])}'
            ax.set_title(title, size=50)
            ax.set_axis_off()

            return ax

        # anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=1000)
        anim = animation.FuncAnimation(fig, partial(animate, images=images, df_scenes=df_scenes),
                                       frames=len(images), interval=1000)
        anim.save(os.path.join(app_folder, 'output', output_name), writer='Pillow')
    except Exception as e:
        print(str(e))
    finally:
        ee.logout()
        for filename in os.listdir(os.path.join(app_folder, 'data')):
            file_path = os.path.join(os.path.join(app_folder, 'data'), filename)
            os.remove(file_path)


if __name__ == "__main__":
    main()
