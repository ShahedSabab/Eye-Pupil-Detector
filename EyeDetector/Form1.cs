using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Collections;
using MetroFramework.Forms;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using System.IO;

namespace EyeDetector
{
    public partial class facescalerateTrack : MetroForm
    {
        private Capture _capture;
        private HaarCascade _faces;
        private HaarCascade _eyes;
        MCvAvgComp[][] facesDetected;
        
        int xMax;
        int yMax;

        double[] scaleRateArray = {1.1,1.2,1.3,1.4}; 

        double faceScaleRate;
        int faceMinNeighbourTheshold;

        double rightScaleRate;
        int rightMinimumNeighbourThreshold;

        double leftScaleRate;
        int leftMinimumNeighbourThreshold;

        int binnaryThresholdR;
        int rightCanny;
        int rightAccumulator;
        int minRadiusR;
        int maxRadiusR;
        int xCoorR;
        int yCoorR;
        double dpR;
        double minDistR;

        int binnaryThresholdL;
        int leftCanny;
        int leftAccumulator;
        int minRadiusL;
        int maxRadiusL;
        int xCoorL;
        int yCoorL;
        double dpL;
        double minDistL;




        int RFixationiteration = 0;
        int LFixationiteration = 0;
        int[] RFixationFrequency = new int[10000];
        int[] LFixationFrequency = new int[10000];
        int LMaxFixation;
        int LMinFixation;
        double LAvgFixation;

        int RMaxFixation;
        int RMinFixation;
        double RAvgFixation;


        int counterRight = 0;
        int counterLeft = 0;
        double horzCounterRight = 0;
        double vartCountRight = 0;
        double horzCounterLeft = 0;
        double vartCountLeft = 0;
        PointF previous_pointRight;
        PointF previous_pointLeft;

        bool duration = false;
        int FixRightHor = 0;
        int FixRightVar = 0;
        int FixLeftHor = 0;
        int FixLeftVar = 0;

        
        Boolean playFlag = true;

        Image<Gray, Byte> grayFrame;
        Image<Gray,Byte> grayFrameEdited;
        Rectangle possibleROI_rightEye;
        Rectangle possibleROI_leftEye;
        MCvAvgComp[][] rightEyesDetected;
        MCvAvgComp[][] leftEyesDetected;
        Point startingPointSearchEyes;
        Point startingLeftEyePointOptimized;
        Image<Bgr, Byte> frame;

        Image<Bgr, Byte> pointFrame;
        Image<Gray, Byte> testDelete;


        bool STopROI = false;
        bool StopFace = false;

        #region Variables
        float px, py, cx, cy, ix, iy;
        float ax, ay;
        #endregion

        #region Kalman Filter and Poins Lists
        PointF[] oup = new PointF[2];
        private Kalman kal;
        private SyntheticData syntheticData;
        private List<PointF> mousePoints;
        private List<PointF> kalmanPoints;
        #endregion

        #region Timers
        Timer MousePositionTaker = new Timer();
        Timer KalmanOutputDisplay = new Timer();
        #endregion

        public facescalerateTrack()
        {
            InitializeComponent();

            KalmanFilter(); //initialize kalman filter

            _capture = new Capture();
            _faces = new HaarCascade("haarcascade_frontalface_alt_tree.xml");
            _eyes = new HaarCascade("haarcascade_eye.xml");

            //face and eye detection

            #region Face and Eye Detection

            faceScaleRate = scaleRateArray[2];    //scaleRateArray = {1.1,1.2,1.3,1.4}
            faceMinNeighbourTheshold = 4;

            rightScaleRate = scaleRateArray[2];
            rightMinimumNeighbourThreshold = 3;

            leftScaleRate = scaleRateArray[2];
            leftMinimumNeighbourThreshold = 3;

            #endregion

      
            faceScaleTrack.Value = Convert.ToInt32(10*(faceScaleRate - 1));
            faceScaleText.Text = faceScaleRate.ToString();        

            faceDetectionTrack.Value = faceMinNeighbourTheshold;
            faceDetectionText.Text = faceDetectionTrack.Value.ToString();

            rightScaleTrack.Value = Convert.ToInt32(10 * (rightScaleRate - 1));
            rightScaleText.Text = rightScaleRate.ToString();

            leftScaleTrack.Value = Convert.ToInt32(10 * (leftScaleRate - 1));
            leftScaleText.Text = leftScaleRate.ToString(); 

            rightDetectionTrack.Value = rightMinimumNeighbourThreshold;
            rightDetectionText.Text = rightDetectionTrack.Value.ToString();

            leftDetectionTrack.Value = leftMinimumNeighbourThreshold;
            leftDetectionText.Text = leftDetectionTrack.Value.ToString();           


            //******face and eye detection******//

            //pupil detection

            #region Pupil Detection

            binnaryThresholdR = 150;
            rightCanny = 200;
            rightAccumulator = 20;
            dpR = 2.0;
            minDistR = 50;
            minRadiusR = 7;
            maxRadiusR = 10;

            binnaryThresholdL = 150;
            leftCanny = 200;
            leftAccumulator = 20;
            dpL = 2.0;
            minDistL = 50.0;
            minRadiusL = 7;
            maxRadiusL = 10;

            #endregion



            rightCannyThreshold.Value = rightCanny;
            rightAccumulatorThreshold.Value = rightAccumulator;
            rightBinnarythreshold.Value = binnaryThresholdR;
            rightDp.Value = (int)dpR;
            rightMinDist.Value = (int)minDistR;
            rightMinRadius.Value = minRadiusR;
            rightMaxRadius.Value = maxRadiusR;

            leftCannyThreshold.Value = leftCanny;
            leftAccumulatorThreshold.Value = leftAccumulator;
            leftBinnarythreshold.Value = binnaryThresholdL;
            leftDp.Value = (int)dpL;
            leftMinDist.Value = (int)minDistL;
            leftMinRadius.Value = minRadiusL;
            leftMaxRadius.Value = maxRadiusL;

            leftBinnarythreshold_text.Text = binnaryThresholdL.ToString();
            rightBinnarythreshold_text.Text = binnaryThresholdR.ToString();

            rightAccumulatorText.Text = rightAccumulatorThreshold.Value.ToString();
            rightCannyText.Text = rightCannyThreshold.Value.ToString();
            rightDpText.Text = rightDp.Value.ToString();
            rightMinDistText.Text = rightMinDist.Value.ToString();
            rightMinRadiusText.Text = rightMinRadius.Value.ToString();
            rightMaxRadiusText.Text = rightMaxRadius.Value.ToString();

            leftAccumulatorText.Text = leftAccumulatorThreshold.Value.ToString();
            leftCannyText.Text = leftCannyThreshold.Value.ToString();
            leftDpText.Text = leftDp.Value.ToString();
            leftMinDistText.Text = leftMinDist.Value.ToString();
            leftMinRadiusText.Text = leftMinRadius.Value.ToString();
            rightMaxRadiusText.Text = rightMaxRadius.Value.ToString();

            for (int i = 0; i < 10000;i++ )
            {
                RFixationFrequency[i] = 0;
                LFixationFrequency[i] = 0;
            }


                //*****pupil detection*****

                Application.Idle += new EventHandler(FrameGrabber);

        }


        private void KalmanFilterRunner(object sender, EventArgs e)
        {
            PointF inp = new PointF(ix, iy);
            oup = new PointF[2];
            oup = filterPoints(inp);
            PointF[] pts = oup;
        }
        public PointF[] filterPoints(PointF pt)
        {
            syntheticData.state[0, 0] = pt.X;
            syntheticData.state[1, 0] = pt.Y;
            Matrix<float> prediction = kal.Predict();
            PointF predictPoint = new PointF(prediction[0, 0], prediction[1, 0]);
            PointF measurePoint = new PointF(syntheticData.GetMeasurement()[0, 0],
                syntheticData.GetMeasurement()[1, 0]);
            Matrix<float> estimated = kal.Correct(syntheticData.GetMeasurement());
            PointF estimatedPoint = new PointF(estimated[0, 0], estimated[1, 0]);
            syntheticData.GoToNextState();
            PointF[] results = new PointF[2];
            results[0] = predictPoint;
            results[1] = estimatedPoint;
            px = predictPoint.X;
            py = predictPoint.Y;
            cx = estimatedPoint.X;
            cy = estimatedPoint.Y;
            return results;
        }
        public void KalmanFilter()
        {
            mousePoints = new List<PointF>();
            kalmanPoints = new List<PointF>();
            kal = new Kalman(4, 2, 0);
            syntheticData = new SyntheticData();
            Matrix<float> state = new Matrix<float>(new float[]
            {
                0.0f, 0.0f, 0.0f, 0.0f
            });
            kal.CorrectedState = state;
            kal.TransitionMatrix = syntheticData.transitionMatrix;
            kal.MeasurementNoiseCovariance = syntheticData.measurementNoise;
            kal.ProcessNoiseCovariance = syntheticData.processNoise;
            kal.ErrorCovariancePost = syntheticData.errorCovariancePost;
            kal.MeasurementMatrix = syntheticData.measurementMatrix;
        }
        private void MousePositionRecord(object sender, EventArgs e)
        {
            Random rand = new Random();
            ix = (int)ax;
            iy = (int)ay;
            //MouseTimed_LBL.Text = "Mouse Position Timed- X:" + ix.ToString() + " Y:" + iy.ToString();
        }

        private void InitialiseTimers(int Timer_Interval = 1000)
        {
            MousePositionTaker.Interval = Timer_Interval;
            MousePositionTaker.Tick += new EventHandler(MousePositionRecord);
            MousePositionTaker.Start();
            KalmanOutputDisplay.Interval = Timer_Interval;
            KalmanOutputDisplay.Tick += new EventHandler(KalmanFilterRunner);
            KalmanOutputDisplay.Start();
        }
        private void StopTimers()
        {
            MousePositionTaker.Tick -= new EventHandler(MousePositionRecord);
            MousePositionTaker.Stop();
            KalmanOutputDisplay.Tick -= new EventHandler(KalmanFilterRunner);
            KalmanOutputDisplay.Stop();
        }

        void FrameGrabber(object sender, EventArgs e)
        {
            //We are acquiring a new frame

          

            //_capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH, 1280);
            //_capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT, 720);

            frame = _capture.QueryFrame().Resize(800, 800, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            //We convert it to grayscale
            grayFrame = frame.Convert<Gray, Byte>();
            //Equalization step
            grayFrame._EqualizeHist();

            // We assume there's only one face in the video
           
            //MCvAvgComp[][] facesDetected = grayFrame.DetectHaarCascade(_faces, faceScaleRate, faceMinNeighbourTheshold, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(frame.Height / 4, frame.Width / 4));
            if (StopFace == false)
            {
                facesDetected = grayFrame.DetectHaarCascade(_faces, faceScaleRate, faceMinNeighbourTheshold, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(25, 25));
            }

            if (facesDetected[0].Length == 1)
            {
                MCvAvgComp face = facesDetected[0][0];
                //Set the region of interest on the faces                

                #region Search Roi based on Face Metric Estimation --- based on empirical measuraments on a couple of photos ---  a really trivial heuristic

                // Our Region of interest where find eyes will start with a sample estimation using face metric

                //Int32 yCoordStartSearchEyes = face.rect.Top + (face.rect.Height * 3 / 12);
                //startingPointSearchEyes = new Point(face.rect.X, yCoordStartSearchEyes);
                //Point endingPointSearchEyes = new Point((face.rect.X + face.rect.Width), yCoordStartSearchEyes);

                //Size searchEyesAreaSize = new Size(face.rect.Width, (face.rect.Height * 2 / 10));
                //Point lowerEyesPointOptimized = new Point(face.rect.X, yCoordStartSearchEyes + searchEyesAreaSize.Height);
                //Size eyeAreaSize = new Size(face.rect.Width / 2, (face.rect.Height * 3 / 10));
                //startingLeftEyePointOptimized = new Point(face.rect.X + face.rect.Width / 2, yCoordStartSearchEyes);

                Int32 yCoordStartSearchEyes = face.rect.Top + (face.rect.Height * 3 / 11);
                startingPointSearchEyes = new Point(face.rect.X, yCoordStartSearchEyes);
                Point endingPointSearchEyes = new Point((face.rect.X + face.rect.Width), yCoordStartSearchEyes);

                Size searchEyesAreaSize = new Size(face.rect.Width, (face.rect.Height * 2 / 9));
                Point lowerEyesPointOptimized = new Point(face.rect.X, yCoordStartSearchEyes + searchEyesAreaSize.Height);
                Size eyeAreaSize = new Size(face.rect.Width / 2, (face.rect.Height * 2 / 9));
                startingLeftEyePointOptimized = new Point(face.rect.X + face.rect.Width / 2, yCoordStartSearchEyes);

                Rectangle possibleROI_eyes = new Rectangle(startingPointSearchEyes, searchEyesAreaSize);
                possibleROI_rightEye = new Rectangle(startingPointSearchEyes, eyeAreaSize);
                possibleROI_leftEye = new Rectangle(startingLeftEyePointOptimized, eyeAreaSize);

                #endregion

                #region Drawing Utilities
                // Let's draw our search area, first the upper line
                frame.Draw(new LineSegment2D(startingPointSearchEyes, endingPointSearchEyes), new Bgr(Color.White), 3);
                // draw the bottom line
                frame.Draw(new LineSegment2D(lowerEyesPointOptimized, new Point((lowerEyesPointOptimized.X + face.rect.Width), (yCoordStartSearchEyes + searchEyesAreaSize.Height))), new Bgr(Color.White), 3);
                // draw the eyes search vertical line
                frame.Draw(new LineSegment2D(startingLeftEyePointOptimized, new Point(startingLeftEyePointOptimized.X, (yCoordStartSearchEyes + searchEyesAreaSize.Height))), new Bgr(Color.White), 3);

                MCvFont font = new MCvFont(FONT.CV_FONT_HERSHEY_TRIPLEX, 0.6d, 0.6d);
                frame.Draw("", ref font, new Point((startingLeftEyePointOptimized.X - 80), (yCoordStartSearchEyes + searchEyesAreaSize.Height + 15)), new Bgr(Color.Yellow));
                frame.Draw("Right Eye", ref font, new Point(startingPointSearchEyes.X, startingPointSearchEyes.Y - 10), new Bgr(Color.Yellow));
                frame.Draw("Left Eye", ref font, new Point(startingLeftEyePointOptimized.X + searchEyesAreaSize.Height / 2, startingPointSearchEyes.Y - 10), new Bgr(Color.Yellow));
                #endregion


                if (STopROI == false)
                {
                    grayFrame.ROI = possibleROI_rightEye;
                    rightEyesDetected = grayFrame.DetectHaarCascade(_eyes, rightScaleRate, rightMinimumNeighbourThreshold, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(60, 60));
                    grayFrame.ROI = Rectangle.Empty;

                    grayFrame.ROI = possibleROI_leftEye;
                    leftEyesDetected = grayFrame.DetectHaarCascade(_eyes, leftScaleRate, leftMinimumNeighbourThreshold, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(60, 60));
                    grayFrame.ROI = Rectangle.Empty;
                }
                //If we are able to find eyes inside the possible face, it should be a face, maybe we find also a couple of eyes
                //if (leftEyesDetected[0].Length != 0 && rightEyesDetected[0].Length != 0)
                //{
                    //draw the face
                    frame.Draw(face.rect, new Bgr(Color.Violet), 2);


                    #region Hough Circles Eye Detection

                    Application.Idle += calcRightEye;

                    Application.Idle += calcLeftEye;            

                    #endregion

                    
                    imageBox1.Image = frame;
           
       
            }
        }
        private void calcRightEye(object sender, EventArgs arg)
        {
     

            //*********
            Graphics G = pictureBox1.CreateGraphics();
            //********
            grayFrame.ROI = possibleROI_rightEye;
            foreach (MCvAvgComp eyeRight in rightEyesDetected[0])
            {
                Rectangle eyeRect = eyeRight.rect;
                eyeRect.Offset(startingPointSearchEyes.X, startingPointSearchEyes.Y);
                frame.Draw(eyeRect, new Bgr(Color.Red), 2);
                grayFrame.ROI = eyeRect;

                //grayFrame = grayFrame.Resize(160, 140, Emgu.CV.CvEnum.INTER.CV_INTER_LINEAR);

               rightE.Image = grayFrame;
               imageBox3.Image = grayFrame;
               //testDelete = grayFrame;
                
               //grayFrame._EqualizeHist();
                grayFrameEdited = grayFrame.Not();
                grayFrameEdited = grayFrameEdited.ThresholdBinary(new Gray(binnaryThresholdR), new Gray(255));
                grayFrameEdited = grayFrameEdited.PyrDown().PyrUp();
          
                //grayFrameEdited = grayFrameEdited.SmoothGaussian(9);
   
     
                //grayFrameEdited.Laplace(1);

              

                rightTrans.Image = grayFrameEdited;
              
                CircleF[] rightEyecircles = grayFrameEdited.HoughCircles(new Gray(rightCanny), new Gray(rightAccumulator), dpR, minDistR, minRadiusR, maxRadiusR)[0];
                //CircleF[] rightEyecircles = grayFrame.HoughCircles(new Gray(rightCanny), new Gray(rightAccumulator), dpR, minDistR, minRadiusR, maxRadiusR)[0];
               
                foreach (CircleF circle in rightEyecircles)
                {
                xCoorR = (int)circle.Center.X;
                yCoorR = (int)circle.Center.Y;

                Point cur_point = new Point(xCoorR, yCoorR);
                PointF pre_point;
                PointF est_point;

                PointF[] results = new PointF[2];
                results = filterPoints(cur_point);
                pre_point = results[0];
                est_point = results[1];
                
                    
                //For Feature Extraction Right Eye//////////////////////////////////////////////////////////////////////////////////////////

                if (duration == true)
                {
                    if (counterRight == 0)
                    {
                        previous_pointRight = est_point;
                        counterRight++;
                    }
                    else
                    {
                        double XCordinateChange = Math.Abs(previous_pointRight.X - est_point.X);
                        double YCordinateChange = Math.Abs(previous_pointRight.Y - est_point.Y);
                        if (XCordinateChange > 2)
                        {
                            horzCounterRight++;
                        }
                        if (YCordinateChange > 1)
                        {
                            vartCountRight++;
                        }
                        if (XCordinateChange > 2 && YCordinateChange > 1)
                        {
                            RFixationiteration++;
                        }
                        if (XCordinateChange < 2 && YCordinateChange > 1)
                        {
                            RFixationiteration++;
                        }
                        if (XCordinateChange > 2 && YCordinateChange < 1)
                        {
                            RFixationiteration++;
                        }

                        if (XCordinateChange < 2 && YCordinateChange < 1)
                        {
                            if (XCordinateChange != 0 && YCordinateChange != 0)
                            {
                                RFixationFrequency[RFixationiteration]++;
                            }
                        }
                        
                        previous_pointRight = est_point;
                        
                    }
                }
                ////////////////////////////////////////////////////////////////////////////////////////////////////////
                


                    //pictureBox1.Refresh();
                    CvInvoke.cvCircle(grayFrame,
                                             new Point((int)est_point.X, (int)est_point.Y),
                                            1,
                                             new MCvScalar(0, 255, 0), //for green colour
                                             -1, //thickness
                                             LINE_TYPE.CV_AA, //for smoothness
                                             0);

                    grayFrame.Draw(circle,
                                   new Gray(255),
                                   2); // thickness

            
                    coordinateR.AppendText("x : " + est_point.X + ",  y : " + est_point.Y + "\n");
                    textBox2.AppendText("x : " + est_point.X + ",  y : " + est_point.Y + "\n"); 

                    Point poi = new Point(xCoorR, yCoorR);

                    G.FillEllipse(Brushes.Magenta, circle.Center.X, circle.Center.Y, 5, 5);
                }

            }
        }

        private void calcLeftEye(object sender, EventArgs arg)
        {

            grayFrame.ROI = possibleROI_leftEye;

            foreach (MCvAvgComp eyeleft in leftEyesDetected[0])
            {
                Rectangle eyeRect = eyeleft.rect;
                eyeRect.Offset(startingLeftEyePointOptimized.X, startingLeftEyePointOptimized.Y);
                frame.Draw(eyeRect, new Bgr(Color.Red), 2);
                grayFrame.ROI = eyeRect;
                //grayFrame._EqualizeHist();
                //grayFrame.SmoothMedian(11);
                //grayFrame.Sobel(1,0,3);
                // grayFrame.Laplace(1);

                leftE.Image = grayFrame;
                imageBox4.Image = grayFrame;
                // grayFrameEdited = grayFrame.Convert<Gray, Byte>().Copy(new Rectangle(10, 30, grayFrame.Width, grayFrame.Height));

                grayFrameEdited = grayFrame.Not();
                grayFrameEdited = grayFrameEdited.ThresholdBinary(new Gray(binnaryThresholdL), new Gray(255));
                grayFrameEdited = grayFrameEdited.PyrDown().PyrUp();

                leftTrans.Image = grayFrameEdited;
                
                
                CircleF[] leftEyecircles = grayFrameEdited.HoughCircles(new Gray(leftCanny), new Gray(leftAccumulator), dpL, minDistL, minRadiusL, maxRadiusL)[0];
                //CircleF[] leftEyecircles = grayFrame.HoughCircles(new Gray(leftCanny), new Gray(leftAccumulator), dpL, minDistL, minRadiusL, maxRadiusL)[0];

                foreach (CircleF circle in leftEyecircles)
                {
                    xCoorL = (int)circle.Center.X;
                    yCoorL = (int)circle.Center.Y;

                    Point cur_point = new Point(xCoorL, yCoorL);
                    PointF pre_point;
                    PointF est_point;

                    PointF[] results = new PointF[2];
                    results = filterPoints(cur_point);
                    pre_point = results[0];
                    est_point = results[1];

                    //For Feature Extraction Right Eye//////////////////////////////////////////////////////////////////////////////////////////

                    if (duration == true)
                    {
                        if (counterLeft == 0)
                        {
                            previous_pointLeft = est_point;
                            counterLeft++;
                        }
                        else
                        {
                            double XCordinateChange = Math.Abs(previous_pointLeft.X - est_point.X);
                            double YCordinateChange = Math.Abs(previous_pointLeft.Y - est_point.Y);
                            if (XCordinateChange > 2)
                            {
                                horzCounterLeft++;
                            }
                            if (YCordinateChange > 1)
                            {
                                vartCountLeft++;
                            }

                            if (XCordinateChange > 2 && YCordinateChange > 1)
                            {
                                LFixationiteration++;
                            }
                            if (XCordinateChange < 2 && YCordinateChange > 1)
                            {
                                LFixationiteration++;
                            }
                            if (XCordinateChange > 2 && YCordinateChange < 1)
                            {
                                LFixationiteration++;
                            }

                            if (XCordinateChange < 2 && YCordinateChange < 1)
                            {
                                if (XCordinateChange != 0 && YCordinateChange != 0)
                                {
                                    LFixationFrequency[LFixationiteration]++;
                                }
                            }
                            previous_pointLeft = est_point;
                        }
                    }
                    ////////////////////////////////////////////////////////////////////////////////////////////////////////

                 
                    CvInvoke.cvCircle(grayFrame,
                                             new Point((int)est_point.X, (int)est_point.Y),
                                            1,
                                             new MCvScalar(0, 255, 0), //for green colour
                                             -1, //thickness
                                             LINE_TYPE.CV_AA, //for smoothness
                                             0);

                    grayFrame.Draw(circle,
                                        new Gray(255),
                                           2); // thickness


                    coordinateL.AppendText("x : " + est_point.X + ",  y : " + est_point.Y + "\n");
                    textBox3.AppendText("x : " + est_point.X + ",  y : " + est_point.Y + "\n"); 
              

                }

                grayFrame.ROI = Rectangle.Empty;
            }
        }

        private void rightCannyThreshold_Scroll(object sender, EventArgs e)
        {
            rightCanny = rightCannyThreshold.Value;
            rightCannyText.Text = rightCannyThreshold.Value.ToString();
        }

        private void rightAccumulatorThreshold_Scroll(object sender, EventArgs e)
        {
            rightAccumulator = rightAccumulatorThreshold.Value;
            rightAccumulatorText.Text = rightAccumulatorThreshold.Value.ToString();
        }

        private void leftCannyThreshold_Scroll(object sender, EventArgs e)
        {
            leftCanny = leftCannyThreshold.Value;
            leftCannyText.Text = leftCannyThreshold.Value.ToString();
        }

        private void leftAccumulatorThreshold_Scroll(object sender, EventArgs e)
        {
            leftAccumulator = leftAccumulatorThreshold.Value;
            leftAccumulatorText.Text = leftAccumulatorThreshold.Value.ToString();
        }

        private void leftCannyText_TextChanged(object sender, EventArgs e)
        {
            leftCannyText.Text = leftCannyThreshold.Value.ToString();
          
        }

        private void leftAccumulatorText_TextChanged(object sender, EventArgs e)
        {
            leftAccumulatorText.Text = leftAccumulatorThreshold.Value.ToString();
        }


        private void button1_Click(object sender, EventArgs e)
        {
            if(playFlag == true)
            {
                playFlag = false;
                button1.Text = "PLAY";
                button7.Text = "PLAY";
                Application.Idle -= new EventHandler(FrameGrabber);
            }
            else 
            {
                playFlag = true;
                button1.Text = "PAUSE";
                button7.Text = "PAUSE";
                Application.Idle += new EventHandler(FrameGrabber);
            }
            
        }

        private void rightBinnarythreshold_Scroll(object sender, EventArgs e)
        {
            binnaryThresholdR = rightBinnarythreshold.Value;
            rightBinnarythreshold_text.Text = binnaryThresholdR.ToString();
        }

        private void leftBinnarythreshold_Scroll(object sender, EventArgs e)
        {
            binnaryThresholdL = leftBinnarythreshold.Value;
            leftBinnarythreshold_text.Text = binnaryThresholdL.ToString();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
        
            mousePoint.Text = "PictureBox- X:" + e.X.ToString() + " Y:" + e.Y.ToString();
        }

        private void rightE_MouseMove(object sender, MouseEventArgs e)
        {

            mousePoint.Text = "Image- X:" + e.X.ToString() + " Y:" + e.Y.ToString();
        }

        private void rightBinnarythreshold_text_Click(object sender, EventArgs e)
        {

        }

        private void rightCannyText_Click(object sender, EventArgs e)
        {

        }

        private void rightAccumulatorText_Click(object sender, EventArgs e)
        {

        }

        private void rightDp_Scroll(object sender, EventArgs e)
        {
            dpR = rightDp.Value;
            rightDpText.Text = rightDp.Value.ToString();
        }

        private void leftDp_Scroll(object sender, EventArgs e)
        {
            dpL = leftDp.Value;
            leftDpText.Text = leftDp.Value.ToString();
        }

        private void rightMinDist_Scroll(object sender, EventArgs e)
        {
            minDistR = rightMinDist.Value;
            rightMinDistText.Text = rightMinDist.Value.ToString();
        }

        private void leftMinDist_Scroll(object sender, EventArgs e)
        {
            minDistL = leftMinDist.Value;
            leftMinDistText.Text = leftMinDist.Value.ToString();
        }

        private void rightMinRadius_Scroll(object sender, EventArgs e)
        {
            minRadiusR = rightMinRadius.Value;
            rightMinRadiusText.Text = rightMinRadius.Value.ToString();
        }

        private void leftMinRadius_Scroll(object sender, EventArgs e)
        {
            minRadiusL = leftMinRadius.Value;
            leftMinRadiusText.Text = leftMinRadius.Value.ToString();
        }

        private void rightMaxRadius_Scroll(object sender, EventArgs e)
        {
            maxRadiusR = rightMaxRadius.Value;
            rightMaxRadiusText.Text = rightMaxRadius.Value.ToString();
        }

        private void leftMaxRadius_Scroll(object sender, EventArgs e)
        {
            maxRadiusL = leftMaxRadius.Value;
            leftMaxRadiusText.Text = leftMaxRadius.Value.ToString();
        }

        private void faceScaleTrack_Scroll(object sender, EventArgs e)
        {
            faceScaleRate = scaleRateArray[faceScaleTrack.Value-1];
            faceScaleText.Text = faceScaleRate.ToString();
        }

        private void faceDetectionTrack_Scroll(object sender, EventArgs e)
        {
            faceMinNeighbourTheshold = faceDetectionTrack.Value;
            faceDetectionText.Text = faceDetectionTrack.Value.ToString();
        }

        private void rightScaleTrack_Scroll(object sender, EventArgs e)
        {
            rightScaleRate = scaleRateArray[rightScaleTrack.Value - 1];
            rightScaleText.Text = rightScaleRate.ToString();
        }


        private void leftScaleTrack_Scroll(object sender, EventArgs e)
        {
            leftScaleRate = scaleRateArray[leftScaleTrack.Value - 1];
            leftScaleText.Text = leftScaleRate.ToString();
        }

        private void rightDetectionTrack_Scroll(object sender, EventArgs e)
        {
            rightMinimumNeighbourThreshold = rightDetectionTrack.Value;
            rightDetectionText.Text = rightDetectionTrack.Value.ToString();
        }


        private void leftDetectionTrack_Scroll(object sender, EventArgs e)
        {
            leftMinimumNeighbourThreshold = leftDetectionTrack.Value;
            leftDetectionText.Text = leftDetectionTrack.Value.ToString();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            InitialiseTimers(10000000);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            StopTimers();
        }

        private void rightE_MouseMove_1(object sender, MouseEventArgs e)
        {
            textBox1.Text = "PictureBox- X:" + e.X.ToString() + " Y:" + e.Y.ToString();
        }

        private void imageBox1_MouseMove(object sender, MouseEventArgs e)
        {
            mousePoint.Text =  "X:" + e.X.ToString() + " Y:" + e.Y.ToString();
        }

        private void button3_Click_1(object sender, EventArgs e)
        {
            metroTextBox1.Visible = true;
            pictureBox2.Visible = false;
            Stream myStream = null;
            OpenFileDialog openFileDialog1 = new OpenFileDialog();

            openFileDialog1.InitialDirectory = "c:\\";
            openFileDialog1.Filter = "txt files (*.txt)|*.txt|All files (*.*)|*.*";
            openFileDialog1.FilterIndex = 2;
            openFileDialog1.RestoreDirectory = true;

            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    if ((myStream = openFileDialog1.OpenFile()) != null)
                    {
                        using (myStream)
                        {
                            metroTextBox1.Text = File.ReadAllText(openFileDialog1.FileName);
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error: Could not read file from disk. Original error: " + ex.Message);
                }
            }
    
        }

        private void button2_Click_1(object sender, EventArgs e)
        {
            metroTextBox1.Visible = false;
            pictureBox2.Visible = true;
            // open file dialog 
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image files (*.jpg, *.jpeg, *.jpe, *.jfif, *.png) | *.jpg; *.jpeg; *.jpe; *.jfif; *.png";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                pictureBox2.ImageLocation = ofd.FileName;
            }

        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (duration == true)
            {
                Fixation_Calculation();
                string lines = "[" + RMaxFixation                              + "," //Max Fixation Right Eye
                                   + LMaxFixation                              + "," //Max Fixation Left Eye
                                   + RMinFixation                              + "," //Min Fixation Right eye
                                   + LMinFixation                              + "," //Min Fixation Left Eye
                                   + RAvgFixation                              + "," //Avg Fixation Right Eye
                                   + LAvgFixation                              + "," //Avg Fixation Right Eye  
                                   + horzCounterRight / vartCountRight         + "," //Rate Right eye
                                   + horzCounterLeft / vartCountLeft           + "]"; //Rate Left Eye

                System.IO.StreamWriter file = new System.IO.StreamWriter("C:\\Users\\Turbomaster\\Desktop\\EyesDetector\\TESTDATA.txt", true);
                file.WriteLine(lines);
                file.WriteLine();
                file.Close();
                button4.Text = "Feature Extraction";


                for (int i = 0; i < 10000; i++)
                {
                    RFixationFrequency[i] = 0;
                    LFixationFrequency[i] = 0;
                }
                RFixationiteration = 0;
                LFixationiteration = 0;
                horzCounterRight = 0;
                vartCountRight = 0;
                horzCounterLeft = 0;
                vartCountLeft = 0;
                counterLeft = 0;
                counterRight = 0;

                duration = false;
            }
            else
            {
                button4.Text = "Stop Extraction";
                duration = true;
            }
        }

        public void Fixation_Calculation()
        {
            LMaxFixation=0;
            LMinFixation = 0;
            LAvgFixation = 0;

            RMaxFixation = 0;
            RMinFixation = 0;
            RAvgFixation = 0;
           

            LMaxFixation = LFixationFrequency.Max();
            LMinFixation = LFixationFrequency.Min();
            for(int i=0;i<LFixationiteration;i++)
            {
                LAvgFixation = LFixationFrequency[i]+LAvgFixation;
            }
            LAvgFixation = LAvgFixation / LFixationiteration;

            RMaxFixation = RFixationFrequency.Max();
            RMinFixation = RFixationFrequency.Min();
            for (int i = 0; i < RFixationiteration; i++)
            {
                RAvgFixation = RFixationFrequency[i] + RAvgFixation;
            }
            RAvgFixation = RAvgFixation / RFixationiteration;
        }

        private void button5_Click(object sender, EventArgs e)
        {
            if(STopROI == false)
            {
                STopROI = true;
            }
            else
            {
                STopROI = false;
            }
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (StopFace == false)
            {
                StopFace = true;
            }
            else
            {
                StopFace = false;
            }
        }


      


    }
}
