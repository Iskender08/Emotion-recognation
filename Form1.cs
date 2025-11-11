using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
//--> Comp vision
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing.Drawing2D; //--> yaxinnan uzaqdan adiyati yoxdu, knopkalarin dizaynidi
using System.Text;

namespace Emotion_Detection_C__
{
    public partial class Form1 : Form
    {
        // Emgunun mehsullari
        private VideoCapture _capture; //--> zaxvat video (eslinde bu upravlyayuside gedir)
        private CascadeClassifier _faceCascade; //--> cascad sohbeti(Xaar mocuzesi)
        private bool _isCameraRunning = false; //--> flaq dlya proverki

        private Image<Bgr, byte> currentFrame; //--> o knopka vaxti, kadr   current-tekusiy
        private Rectangle lastFace = Rectangle.Empty; //-->kaskadin verdiyi uzun axrinci kordinati(knopka sohbeti burdada kecir)

        public Form1()
        {
            InitializeComponent();
            _faceCascade = new CascadeClassifier(
                Path.Combine(Application.StartupPath, "emotion_project", "haarcascade_frontalface_default.xml"));
            //-->modelin zaqruzkasi(papka->em.py)
            

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            comboBoxCameras.Items.Add("Daxili");
            comboBoxCameras.Items.Add("Xarici");
            comboBoxCameras.SelectedIndex = 0;

            StyleButton(btnStart, "camera_icon.png", btnStart_Paint);
            StyleButton(btnDetect, "detect_icon.gif", btnDetect_Paint);
            StyleButton(btnExit, "exit.png", btnExit_Paint);
            StyleButton(btnStop, "stop.png", btnStop_Paint);

            pictureBoxLoading.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxLoading.Visible = false;
        }
        //-->knopka dizayni
        private void StyleButton(Button button, string imagePath, PaintEventHandler paintHandler)
        {
            button.Size = new Size(60, 60);
            button.BackColor = Color.White;
            button.FlatStyle = FlatStyle.Flat;
            button.FlatAppearance.BorderSize = 0;
            button.Image = Image.FromFile(imagePath);
            button.ImageAlign = ContentAlignment.MiddleCenter;
            button.Text = "";
            button.Paint += paintHandler;
        }

        private void btnStart_Paint(object sender, PaintEventArgs e)
        {
            RoundButton((Button)sender);
        }
        private void btnStop_Paint(object sender, PaintEventArgs e)
        {
            RoundButton((Button)sender);
        }
        private void btnDetect_Paint(object sender, PaintEventArgs e)
        {
            RoundButton((Button)sender);
        }

        private void btnExit_Paint(object sender, PaintEventArgs e)
        {
            RoundButton((Button)sender);
        }

        private void RoundButton(Button btn)
        {
            GraphicsPath path = new GraphicsPath();
            path.AddEllipse(0, 0, btn.Width, btn.Height);
            btn.Region = new Region(path);
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            //--> birinci yoxla gor varmi kamera
            if (!_isCameraRunning) 
            {
                this.Invoke((MethodInvoker)(() =>
                {
                    pictureBoxLoading.Image = Image.FromFile(Path.Combine(Application.StartupPath, "emotion_project", "loading.gif"));
                    pictureBoxLoading.Visible = true;
                    lblEmotion.Text = "Kamera açılır...";
                }));

                Task.Run(() =>
                {
                    try
                    {
                        int selectedCamera = comboBoxCameras.SelectedIndex;
                        _capture = new VideoCapture(selectedCamera);
                        _capture.ImageGrabbed += ProcessFrame; //--> +=podpiskadi(ele bil bu obrabotcik sobitiydi), bu bizim platformdu qaqaw
                        _capture.Start();

                        this.Invoke((MethodInvoker)(() =>
                        {
                            _isCameraRunning = true;
                            pictureBoxLoading.Visible = false;
                            lblEmotion.Text = "Kamera hazırdır.";
                        }));
                    }
                    catch (Exception ex)
                    {
                        this.Invoke((MethodInvoker)(() =>
                        {
                            pictureBoxLoading.Visible = false;
                            MessageBox.Show("Kamerada səhv var: " + ex.Message);
                        }));
                    }
                });
            }
        }


        private void ProcessFrame(object sender, EventArgs e)
        {
            try
            {
                Mat frame = new Mat(); //emgunun gozellikleri(kadr inisializasiyasi(obyektdi esli cto))
                _capture.Retrieve(frame);
                currentFrame = frame.ToImage<Bgr, byte>();
                //kadr tapmasa qaytar geri qoy virde axtarsin)))))))))
                if (currentFrame == null) // -->tekusi kadr sohbeti
                    return;
                //--> burani yadda saxlamaq lazimdi, cunki sirf seriy sekilleri gorur bu kaskad
                var gray = currentFrame.Convert<Gray, byte>(); //--> bez problem
                //uz s pomosyu detectmultiscale
                var faces = _faceCascade.DetectMultiScale(gray, 1.1, 4); 
                //soxraneniye posledneqo kordinata
                lastFace = Rectangle.Empty; 

                if (faces.Length > 0)//--> izobrajeniye varsa bomba kimi
                {
                    lastFace = faces[0]; //-->1 doy 0di
                    currentFrame.Draw(lastFace, new Bgr(Color.Green), 2); //kvadrat
                    var faceImage = gray.Copy(lastFace).Resize(200, 200, Inter.Cubic);
                    pictureBoxFace.Image = faceImage.ToBitmap();
                }

                pictureBoxVideo.Image = currentFrame.ToBitmap(); //-->bitmap preobrazovaniye(menasiz)
            }
            catch (Exception ex)
            {
                Console.WriteLine("Kamerada səhv var: " + ex.Message); // -->exeption sehvinde(cixan mesaj)
            }
        }

        private void btnDetect_Click(object sender, EventArgs e)
        {
            if (currentFrame == null || lastFace == Rectangle.Empty) // muqahise etapi 
            {
                MessageBox.Show("Üz tapılmadı");
                return;
            }

            pictureBoxLoading.Image = Image.FromFile(Path.Combine(Application.StartupPath, "emotion_project", "loading.gif"));
            pictureBoxLoading.Visible = true;
            lblEmotion.Text = "Yüklənir...";

            Task.Run(() =>
            {
                try
                {
                    var faceImage = currentFrame.Copy(lastFace).Convert<Gray, byte>().Resize(48, 48, Inter.Cubic);
                    string tempPath = Path.Combine(Application.StartupPath, "temp.jpg");
                    faceImage.Save(tempPath);

                    string emotion = RunPythonAndGetEmotion(tempPath);

                    this.Invoke((MethodInvoker)(() =>
                    {
                        pictureBoxLoading.Visible = false;
                        lblEmotion.Text = "Emosiya: " + emotion;
                    }));
                }
                catch (Exception ex)
                {
                    this.Invoke((MethodInvoker)(() =>
                    {
                        pictureBoxLoading.Visible = false;
                        MessageBox.Show("Emosiyanı hesablamaq mümkün olmadı: " + ex.Message);
                    }));
                }
            });
            
        }

        private string RunPythonAndGetEmotion(string imagePath)
        {
            try
            {
                string scriptPath = Path.Combine(Application.StartupPath, "emotion_project", "detect_emotion.py");
                string modelPath = Path.Combine(Application.StartupPath, "emotion_project", "facialemotionmodel.h5");

                if (!File.Exists(scriptPath) || !File.Exists(imagePath) || !File.Exists(modelPath))
                {
                    return "Fayl tapılmadı";
                }

                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{scriptPath}\" \"{imagePath}\" \"{modelPath}\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    StandardOutputEncoding = Encoding.UTF8 // Az dili herfleri
                };


                using (var process = Process.Start(psi))
                {
                    string output = process.StandardOutput.ReadToEnd().Trim();
                    string error = process.StandardError.ReadToEnd().Trim();
                    process.WaitForExit();

                    if (!string.IsNullOrWhiteSpace(output))
                        return output;

                    if (!string.IsNullOrWhiteSpace(error))
                        MessageBox.Show("Python erroru:\n" + error);

                    return "Tanımadım";
                }
            }
            catch (Exception ex)
            {
                return "proyekt səhvi: " + ex.Message;
            }
        }

        private void btnExit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void pictureBoxFace_Click(object sender, EventArgs e)
        {
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            if (_isCameraRunning)
            {
                _capture.Stop();
                _capture.Dispose();
                _capture = null;
                _isCameraRunning = false;

                lblEmotion.Text = "Kamera dayandı.";
                pictureBoxVideo.Image = null;
                pictureBoxFace.Image = null;
            }
        }
    }
}
