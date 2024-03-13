import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
import sqlite3
import datetime
import sys

# Signal Processing Functions
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def fm_demodulate(fm_signal, fs):
    phase = np.unwrap(np.angle(fm_signal))  # Unwrap the phase of the complex signal
    demodulated = np.diff(phase) / (2 * np.pi) * fs
    return demodulated

# Database Functions
def log_data_to_db(db_conn, data, demodulated_message):
    timestamp = datetime.datetime.now()
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO signals (timestamp, raw_data, message) VALUES (?, ?, ?)",
                   (timestamp, data, demodulated_message))
    db_conn.commit()

# GUI Class
class RadioInterface(QMainWindow):
    def __init__(self, processing_thread):
        super().__init__()
        self.processing_thread = processing_thread
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Radio Signal Interface')
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        self.button = QPushButton('Start', self)
        self.button.clicked.connect(self.start_processing)
        layout.addWidget(self.button)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        self.show()

    def start_processing(self):
        if not self.processing_thread.isRunning():
            self.processing_thread.start()  # Start the thread
            self.button.setText('Stop')
            self.button.clicked.disconnect()  # Disconnect the current clicked signal
            self.button.clicked.connect(self.stop_processing)  # Connect to stop_processing method
        else:
            self.stop_processing()

    def stop_processing(self):
        if self.processing_thread.isRunning():
            self.processing_thread.stop()  # Method to stop the thread safely
            self.processing_thread.wait()  # Wait for the thread to finish
            self.button.setText('Start')
            self.button.clicked.disconnect()  # Disconnect the current clicked signal
            self.button.clicked.connect(self.start_processing)  # Reconnect to start_processing method

    def update_spectrum(self, spectrum):
        self.ax.clear()
        self.ax.plot(spectrum)
        self.canvas.draw()

# Real-time Processing Thread
class ProcessingThread(QThread):
    new_spectrum = pyqtSignal(np.ndarray)

    def __init__(self, sdr_settings):
        super().__init__()
        self.sdr_settings = sdr_settings
        self.running = True

    def run(self):
        sdr = RtlSdr()
        sdr.sample_rate = self.sdr_settings['sample_rate']
        sdr.center_freq = self.sdr_settings['center_freq']
        sdr.freq_correction = self.sdr_settings['freq_correction']
        sdr.gain = self.sdr_settings['gain']

        while self.running:
            samples = sdr.read_samples(256*1024)
            filtered = butter_lowpass_filter(samples, 100e3, sdr.sample_rate)
            demodulated = fm_demodulate(filtered, sdr.sample_rate)
            self.new_spectrum.emit(demodulated)  # Emit the demodulated signal for display

        sdr.close()

    def stop(self):
        self.running = False

# Main Function
def main():
    app = QApplication(sys.argv)

    db_conn = sqlite3.connect('radio_data.db')
    db_conn.execute('''CREATE TABLE IF NOT EXISTS signals
                       (id INTEGER PRIMARY KEY,
                        timestamp DATETIME,
                        raw_data BLOB,
                        message TEXT)''')

    sdr_settings = {
        'sample_rate': 2.048e6,  # Hz
        'center_freq': 99.5e6,   # Hz
        'freq_correction': 60,   # PPM
        'gain': 'auto'
    }

    processing_thread = ProcessingThread(sdr_settings)
    gui = RadioInterface(processing_thread)  # Pass the thread to the GUI
    processing_thread.new_spectrum.connect(gui.update_spectrum)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
