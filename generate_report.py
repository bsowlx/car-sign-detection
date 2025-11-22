import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    path = 'runs/detect/train/results.csv'
    
    if not os.path.exists(path):
        print("Could not find results.csv")
        return

    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    if not os.path.exists('report_images'):
        os.mkdir('report_images')

    plt.figure()
    plt.plot(data['epoch'], data['train/box_loss'], label='Train Loss')
    plt.plot(data['epoch'], data['val/box_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('report_images/loss.png')
    
    plt.figure()
    plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP@50')
    plt.title('Mean Average Precision')
    plt.legend()
    plt.savefig('report_images/map.png')

    print("Done. Check report_images folder.")

if __name__ == "__main__":
    main()
