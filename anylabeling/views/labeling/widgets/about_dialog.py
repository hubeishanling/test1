from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
)
from PyQt5.QtCore import Qt, pyqtSignal

from anylabeling.app_info import __version__
from anylabeling.views.labeling.utils.general import open_url
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.widgets.popup import Popup


class AboutDialog(QDialog):
    update_available = pyqtSignal(dict)
    no_update = pyqtSignal()
    error = pyqtSignal(str)

    company_name = "湖北闪灵科技有限公司"
    qq_group = "726703994"
    website_url = "http://www.sanguoyr.top"
    source_url = "https://github.com/CVHub520/X-AnyLabeling"
    description = "免费的脚本开发平台"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.setWindowTitle(" ")
        self.setFixedSize(350, 340)

        self.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
                border-radius: 10px;
            }
            QLabel {
                color: #1d1d1f;
            }
            QPushButton {
                border: none;
                background: transparent;
                color: #0066FF;
                text-align: center;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #F0F0F0;
                border-radius: 4px;
            }
            QPushButton#link-btn {
                color: #0066FF;
            }
            QPushButton#social-btn {
                padding: 8px;
            }
            QPushButton#social-btn:hover {
                background-color: #F0F0F0;
                border-radius: 4px;
            }
            QPushButton#close-btn {
                color: #86868b;
                font-size: 16px;
                padding: 8px;
            }
            QPushButton#close-btn:hover {
                background-color: #F0F0F0;
                border-radius: 4px;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # App name and version
        title_label = QLabel(f"<b>X-AnyLabeling</b> v{__version__}")
        title_label.setStyleSheet("font-size: 16px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(self.description)
        desc_label.setStyleSheet("font-size: 13px; color: #666;")
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Company info
        company_label = QLabel(f"公司：{self.company_name}")
        company_label.setStyleSheet("font-size: 13px;")
        company_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(company_label)

        # QQ Group
        qq_label = QLabel(f"QQ群：{self.qq_group}")
        qq_label.setStyleSheet("font-size: 13px;")
        qq_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(qq_label)

        # Source project
        source_label = QLabel(f"源项目：{self.source_url}")
        source_label.setStyleSheet("font-size: 11px; color: #888;")
        source_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(source_label)

        # Links row - centered
        links_layout = QHBoxLayout()
        links_layout.setSpacing(8)
        links_layout.setAlignment(Qt.AlignCenter)

        website_btn = QPushButton(self.tr("官网"))
        website_btn.setObjectName("link-btn")
        website_btn.clicked.connect(lambda: open_url(self.website_url))

        copy_btn = QPushButton(self.tr("复制官网地址"))
        copy_btn.setObjectName("link-btn")
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.website_url))

        copy_qq_btn = QPushButton(self.tr("复制QQ群号"))
        copy_qq_btn.setObjectName("link-btn")
        copy_qq_btn.clicked.connect(lambda: self.copy_to_clipboard(self.qq_group))

        links_layout.addWidget(website_btn)
        links_layout.addWidget(QLabel("·"))
        links_layout.addWidget(copy_btn)
        links_layout.addWidget(QLabel("·"))
        links_layout.addWidget(copy_qq_btn)
        layout.addLayout(links_layout)

        # Source project link
        source_layout = QHBoxLayout()
        source_layout.setAlignment(Qt.AlignCenter)
        source_btn = QPushButton(self.tr("访问源项目"))
        source_btn.setObjectName("link-btn")
        source_btn.clicked.connect(lambda: open_url(self.source_url))
        source_layout.addWidget(source_btn)
        layout.addLayout(source_layout)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Copyright
        copyright_label = QLabel(
            f"Copyright © 2023 {self.company_name}. All rights reserved."
        )
        copyright_label.setStyleSheet("color: #86868b; font-size: 12px;")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)

        self.move_to_center()

    def move_to_center(self):
        """Move dialog to center of the screen"""
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def copy_to_clipboard(self, text):
        """Copy text to clipboard and show popup"""
        popup = Popup(
            self.tr("Copied!"),
            self.parent,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self.parent, copy_msg=text)
