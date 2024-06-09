using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.IO;
using System.Linq;
using System.Windows;
using Microsoft.Win32;
using System.Windows.Media.Imaging;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FoodClassifier
{
    public partial class MainWindow : Window
    {
        private MLContext mlContext;
        private ITransformer? mlModel;
        private string modelPath = "FoodModel.zip";

        public MainWindow()
        {
            InitializeComponent();
            mlContext = new MLContext();
            LoadModel();
        }

        private void LoadModel()
        {
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }
        }

        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Image files (*.png;*.jpeg)|*.png;*.jpeg|All files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                var bitmap = new BitmapImage(new Uri(openFileDialog.FileName));
                ImageDisplay.Source = bitmap;
                ImageFilePath = openFileDialog.FileName;
            }
        }

        private void ClassifyImage_Click(object sender, RoutedEventArgs e)
        {
            if (!string.IsNullOrEmpty(ImageFilePath))
            {
                var prediction = PredictImage(ImageFilePath);
                ResultText.Text = $"Продукт питания: {prediction.PredictedLabel} с вероятностью {prediction.Score.Max():P2}";
            }
            else
            {
                MessageBox.Show("Пожалуйста, загрузите изображение перед классификацией.");
            }
        }

        private string? ImageFilePath { get; set; }

        public class ImageInput
        {
            [ImageType(200, 200)]
            public byte[] Image { get; set; } = Array.Empty<byte>();
        }

        public class ImagePrediction : ImageInput
        {
            public string PredictedLabel { get; set; } = string.Empty;
            public float[] Score { get; set; } = Array.Empty<float>();
        }

        private ImagePrediction PredictImage(string imagePath)
        {
            if (mlModel == null)
            {
                throw new InvalidOperationException("Модель не загружена.");
            }

            var imageBytes = LoadImageAsByteArray(imagePath);

            var imageData = new ImageInput()
            {
                Image = imageBytes
            };

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageInput, ImagePrediction>(mlModel);
            var prediction = predictionEngine.Predict(imageData);

            return prediction;
        }

        private byte[] LoadImageAsByteArray(string imagePath)
        {
            using (var image = Image.Load<Rgb24>(imagePath))
            {
                image.Mutate(x => x.Resize(200, 200));
                using (var ms = new MemoryStream())
                {
                    image.SaveAsBmp(ms);
                    return ms.ToArray();
                }
            }
        }
    }
}
