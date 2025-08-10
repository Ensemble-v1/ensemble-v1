# Ensemble: Sheet Music to MIDI Converter

## Project Overview

Ensemble is a powerful tool that converts images of sheet music into MIDI files. This project leverages state-of-the-art deep learning models and computer vision techniques to accurately recognize and transcribe musical symbols. The system is built with a Flask backend, providing a simple API for users to upload sheet music images and receive MIDI transcriptions.

## Features

- **Sheet Music Analysis:** Analyzes images of sheet music to detect and classify various musical symbols.
- **MIDI Generation:** Converts the detected symbols into a standard MIDI file, which can be played on any MIDI-compatible device or software.
- **YOLOv8 Integration:** Utilizes a custom-trained YOLOv8 model for robust and accurate symbol detection.
- **Flask API:** Provides a simple and easy-to-use REST API for interacting with the system.
- **Extensible and Modular:** The project is designed with a modular architecture, making it easy to extend and improve.

## How It Works

The system works in the following steps:

1.  **Image Upload:** The user uploads an image of sheet music through the API.
2.  **Image Preprocessing:** The uploaded image is preprocessed to enhance its quality and prepare it for analysis.
3.  **Staff Line Detection:** The system detects the staff lines in the image to establish a reference for symbol positioning.
4.  **Symbol Detection:** A custom-trained YOLOv8 model is used to detect and classify musical symbols in the image.
5.  **Symbol Processing:** The detected symbols are processed to extract their musical properties, such as pitch and duration.
6.  **MIDI Generation:** A MIDI file is generated based on the processed symbols.
7.  **API Response:** The API returns a link to the generated MIDI file, along with other analysis details.

## Project Structure

The project is organized into the following directories and files:

-   `app.py`: The main Flask application file, containing the API endpoints and core logic.
-   `model_training.py`: The script for training the YOLOv8 model on the musical symbols dataset.
-   `config.py`: The configuration file for the Flask application.
-   `deepscores.yaml`: The dataset configuration file for YOLOv8.
-   `requirements.txt`: A list of the Python dependencies required to run the project.
-   `static/`: A directory for storing static files, such as uploaded images and generated MIDI files.
-   `utils/`: A directory containing utility modules for image processing, MIDI generation, and other tasks.
-   `datasets/`: A directory containing the musical symbols dataset used for training the model.
-   `models/`: A directory for storing the trained YOLOv8 model.
-   `tests/`: A directory containing unit tests for the project.

## Getting Started

To get started with the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shornalore/sense-civic-scraper-local
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    python app.py
    ```
4.  **Use the API:**
    -   Send a POST request to `/api/analyze` with an image of sheet music to receive a MIDI transcription.

## Model Training

To train the model, you will need to prepare a dataset of musical symbols and then run the `model_training.py` script. The dataset should be organized into the following structure:

```
datasets/
  train/
    images/
    labels/
  val/
    images/
    labels/
```

Once the dataset is prepared, you can run the training script:

```bash
python model_training.py
```

The trained model will be saved in the `models/` directory.

## Contributing

Contributions to the project are welcome! If you would like to contribute, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
