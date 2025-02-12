# TDMS Signal Visualizer

This Python application provides a user-friendly interface for visualizing data stored in TDMS files, commonly used for storing time-series data. Leveraging libraries like `numpy`, `pygame`, `scipy`, and `nptdms`, this tool allows users to interactively explore multi-channel data with features like zooming, panning, and adjustable time windows.

## Key Features:

*   **TDMS Data Loading:** Reads and processes TDMS files, extracting channel names and data for visualization.
*   **Interactive Visualization:** Uses `pygame` for creating a dynamic and responsive user interface.
*   **Multi-Channel Support:** Displays multiple channels simultaneously or focuses on a single channel for detailed analysis.
*   **Zooming and Panning:** Enables users to zoom in/out on the data and pan through the time series to examine specific segments.
*   **Adjustable Time Window (Tao):** Allows modification of the displayed time window for different levels of detail.
*   **Data Filtering:** Implements Gaussian filtering using `scipy.ndimage` to smooth data for better visualization, particularly at higher zoom levels, optimizing performance through precomputed filter caches.
*   **Dynamic Layout:** Adapts to different window sizes, ensuring a consistent and usable experience.
*   **Customizable UI:** Includes a sidebar with interactive buttons and informative tooltips.
*   **Vertical Scaling:** Dynamically adjust the vertical scale of the data for optimal visualization.
![image](https://github.com/user-attachments/assets/a7f251cc-a3cc-41f2-81e0-c561d6aceca0)

## Libraries Used:

*   `numpy`: For numerical computations and data handling.
*   `pygame`: For creating the graphical user interface.
*   `scipy`: For signal processing, specifically Gaussian filtering.
*   `nptdms`: For reading and parsing TDMS files.

## Usage:

1.  **Installation:** Install the required libraries using pip:

    ```bash
    pip install numpy pygame scipy nptdms
    ```

2.  **Configuration:** Modify the `DATA_PATH` variable in the script to point to your TDMS file.

3.  **Execution:** Run the Python script:

    ```bash
    python your_script_name.py
    ```

## Controls:

*   **Zoom In:** Mouse Wheel Up / `+` key
*   **Zoom Out:** Mouse Wheel Down / `-` key
*   **Pan Left:** Left Arrow Key
*   **Pan Right:** Right Arrow Key
*   **Increase Time Window (Tao):** Up Arrow Key
*   **Decrease Time Window (Tao):** Down Arrow Key
*   **Toggle Multi-View:** `M` Key
*   **Previous Channel (Single-Channel Mode):** Page Up Key
*   **Next Channel (Single-Channel Mode):** Page Down Key
*   **Increase Vertical Scale:** `W` Key
*   **Decrease Vertical Scale:** `S` Key


