# Machine Learning for IoT Emergency sirens detection

Development of a complete IoT application powered by Machine Learning to detect and report to the user the presence of an ambulance in the context of a busy road. The main project steps are:
- Audio Processing trough resampling technique, padding, Discrete Fourier Transform and Mel-Frequency Cepstral Coefficients
- Construction of a Convolutional Neural Network to identify emergency sirens in city traffic
- Communication of the results obtained by the network through the MQTT and REST protocols used to connect the devices to the Cloud
- Visualization of the emergency sirens in real-time on a map of Turin and the principal statistics about the location and the period of the emergency sirens in the city
