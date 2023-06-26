import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def explore_plot(df):
    df.plot(subplots=True,figsize=(25,10))
    plt.show()


def plot_true_pred(result_df, y_test, y_pred):
    plt.figure(figsize=(25,10))
    plt.plot(result_df[y_test], marker='.', label="True")
    plt.plot(result_df[y_pred], label="Prediction")
    plt.ylabel('Magnitude')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()


def plot_loss(history):
    plt.figure(figsize=(25,10))
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.legend()
    plt.show()