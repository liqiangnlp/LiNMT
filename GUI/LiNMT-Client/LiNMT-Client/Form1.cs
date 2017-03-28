using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;



using System.Net;
using System.Net.Sockets;
using System.Threading; 



namespace LiNMT_Client
{
    public partial class LiNMT : Form
    {
        public LiNMT()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void translateButton_Click(object sender, EventArgs e)
        {
            outputBox.Text = "";

            string tmpText = inputBox.Text;
            tmpText = tmpText.Replace("\r\n", "");
            tmpText = tmpText.Replace(" ", "");

            if (tmpText == "")
            {
                MessageBox.Show("Input is empty!");
                return;
            }

            IPAddress ip = IPAddress.Parse("10.119.186.29");
            int portNum;
            if (pt2ch.Checked)
            {
                portNum = 8088;
                //portNum = 9999;
            }
            else
            {
                portNum = 8090;
            }

            Socket clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

            try 
            { 
                clientSocket.Connect(new IPEndPoint(ip, portNum));
            }
            catch
            {
                MessageBox.Show("Connecting server failed!");
                return;
            }

            try
            {
                string sendMessage = inputBox.Text;

                //string[] sendMessages = sendMessage.Split(new String[] {"\r\n"}, StringSplitOptions.None);

                //if (sendMessages.Length > 0) {
                //    sendMessage = sendMessages[0];
                //}

                clientSocket.Send(Encoding.UTF8.GetBytes(sendMessage));
                //Thread.Sleep(500);
                byte[] result = new byte[1024];
                int receiveLength = clientSocket.Receive(result);

                outputBox.Text += Encoding.UTF8.GetString(result, 0, receiveLength);
                outputBox.Text = outputBox.Text.Replace("\n", "\r\n");
                outputBox.Text += "\r\n";
                
            }
            catch {
                MessageBox.Show("Connecting server failed!");
                clientSocket.Shutdown(SocketShutdown.Both);
                clientSocket.Close();
                return;
            }
        }

        private void clearButton_Click(object sender, EventArgs e)
        {
            inputBox.Text = "";
            outputBox.Text = "";
        }

        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void inputBox_TextChanged(object sender, EventArgs e)
        {

        }

        private void inputBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Modifiers == Keys.Control && e.KeyCode == Keys.A)
            {
                ((TextBox)sender).SelectAll();
            }  
}

        private void copyAllButton_Click(object sender, EventArgs e)
        {
            inputBox.SelectAll();
            inputBox.Copy();
        }

        private void copyOutputButton_Click(object sender, EventArgs e)
        {
            outputBox.SelectAll();
            outputBox.Copy();
        }
    }
}
