namespace LiNMT_Client
{
    partial class LiNMT
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.inputBox = new System.Windows.Forms.TextBox();
            this.translateButton = new System.Windows.Forms.Button();
            this.outputBox = new System.Windows.Forms.TextBox();
            this.clearButton = new System.Windows.Forms.Button();
            this.pt2ch = new System.Windows.Forms.RadioButton();
            this.ch2pt = new System.Windows.Forms.RadioButton();
            this.copyAllButton = new System.Windows.Forms.Button();
            this.copyOutputButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // inputBox
            // 
            this.inputBox.Font = new System.Drawing.Font("Constantia", 14F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.inputBox.Location = new System.Drawing.Point(30, 93);
            this.inputBox.Multiline = true;
            this.inputBox.Name = "inputBox";
            this.inputBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.inputBox.Size = new System.Drawing.Size(493, 362);
            this.inputBox.TabIndex = 0;
            this.inputBox.TextChanged += new System.EventHandler(this.inputBox_TextChanged);
            // 
            // translateButton
            // 
            this.translateButton.BackColor = System.Drawing.SystemColors.Highlight;
            this.translateButton.Font = new System.Drawing.Font("Buxton Sketch", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.translateButton.ForeColor = System.Drawing.SystemColors.ControlLightLight;
            this.translateButton.Location = new System.Drawing.Point(539, 48);
            this.translateButton.Name = "translateButton";
            this.translateButton.Size = new System.Drawing.Size(116, 28);
            this.translateButton.TabIndex = 1;
            this.translateButton.Text = "Translate";
            this.translateButton.UseVisualStyleBackColor = false;
            this.translateButton.Click += new System.EventHandler(this.translateButton_Click);
            // 
            // outputBox
            // 
            this.outputBox.Font = new System.Drawing.Font("Constantia", 14F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.outputBox.Location = new System.Drawing.Point(539, 93);
            this.outputBox.Multiline = true;
            this.outputBox.Name = "outputBox";
            this.outputBox.ReadOnly = true;
            this.outputBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.outputBox.Size = new System.Drawing.Size(493, 362);
            this.outputBox.TabIndex = 2;
            // 
            // clearButton
            // 
            this.clearButton.BackColor = System.Drawing.SystemColors.Highlight;
            this.clearButton.Font = new System.Drawing.Font("Buxton Sketch", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.clearButton.ForeColor = System.Drawing.SystemColors.ControlLightLight;
            this.clearButton.Location = new System.Drawing.Point(671, 48);
            this.clearButton.Name = "clearButton";
            this.clearButton.Size = new System.Drawing.Size(116, 28);
            this.clearButton.TabIndex = 3;
            this.clearButton.Text = "Clear";
            this.clearButton.UseVisualStyleBackColor = false;
            this.clearButton.Click += new System.EventHandler(this.clearButton_Click);
            // 
            // pt2ch
            // 
            this.pt2ch.AutoSize = true;
            this.pt2ch.Checked = true;
            this.pt2ch.Font = new System.Drawing.Font("Constantia", 10.5F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.pt2ch.Location = new System.Drawing.Point(30, 21);
            this.pt2ch.Name = "pt2ch";
            this.pt2ch.Size = new System.Drawing.Size(256, 30);
            this.pt2ch.TabIndex = 4;
            this.pt2ch.TabStop = true;
            this.pt2ch.Text = "Portuguese to Chinese";
            this.pt2ch.UseVisualStyleBackColor = true;
            // 
            // ch2pt
            // 
            this.ch2pt.AutoSize = true;
            this.ch2pt.Font = new System.Drawing.Font("Constantia", 10.5F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.ch2pt.Location = new System.Drawing.Point(30, 48);
            this.ch2pt.Name = "ch2pt";
            this.ch2pt.Size = new System.Drawing.Size(256, 30);
            this.ch2pt.TabIndex = 5;
            this.ch2pt.Text = "Chinese to Portuguese";
            this.ch2pt.UseVisualStyleBackColor = true;
            // 
            // copyAllButton
            // 
            this.copyAllButton.BackColor = System.Drawing.SystemColors.Highlight;
            this.copyAllButton.Font = new System.Drawing.Font("Buxton Sketch", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.copyAllButton.ForeColor = System.Drawing.SystemColors.ControlLightLight;
            this.copyAllButton.Location = new System.Drawing.Point(30, 461);
            this.copyAllButton.Name = "copyAllButton";
            this.copyAllButton.Size = new System.Drawing.Size(101, 25);
            this.copyAllButton.TabIndex = 6;
            this.copyAllButton.Text = "Copy Input";
            this.copyAllButton.UseVisualStyleBackColor = false;
            this.copyAllButton.Click += new System.EventHandler(this.copyAllButton_Click);
            // 
            // copyOutputButton
            // 
            this.copyOutputButton.BackColor = System.Drawing.SystemColors.Highlight;
            this.copyOutputButton.Font = new System.Drawing.Font("Buxton Sketch", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.copyOutputButton.ForeColor = System.Drawing.SystemColors.ControlLightLight;
            this.copyOutputButton.Location = new System.Drawing.Point(539, 461);
            this.copyOutputButton.Name = "copyOutputButton";
            this.copyOutputButton.Size = new System.Drawing.Size(101, 25);
            this.copyOutputButton.TabIndex = 7;
            this.copyOutputButton.Text = "Copy Output";
            this.copyOutputButton.UseVisualStyleBackColor = false;
            this.copyOutputButton.Click += new System.EventHandler(this.copyOutputButton_Click);
            // 
            // LiNMT
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(10F, 18F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.AliceBlue;
            this.ClientSize = new System.Drawing.Size(1062, 521);
            this.Controls.Add(this.copyOutputButton);
            this.Controls.Add(this.copyAllButton);
            this.Controls.Add(this.ch2pt);
            this.Controls.Add(this.pt2ch);
            this.Controls.Add(this.clearButton);
            this.Controls.Add(this.outputBox);
            this.Controls.Add(this.translateButton);
            this.Controls.Add(this.inputBox);
            this.Font = new System.Drawing.Font("Times New Roman", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.Name = "LiNMT";
            this.Text = "LiNMT Client v1.0";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox inputBox;
        private System.Windows.Forms.Button translateButton;
        private System.Windows.Forms.TextBox outputBox;
        private System.Windows.Forms.Button clearButton;
        private System.Windows.Forms.RadioButton pt2ch;
        private System.Windows.Forms.RadioButton ch2pt;
        private System.Windows.Forms.Button copyAllButton;
        private System.Windows.Forms.Button copyOutputButton;
    }
}

