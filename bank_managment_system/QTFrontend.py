
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import backend
backend.connect_database()
def create_styled_frame(parent, min_size=None, style=""):
    """Create a styled QFrame with optional minimum size and custom style."""
    frame = QtWidgets.QFrame(parent)
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    frame.setFrameShadow(QtWidgets.QFrame.Raised)
    if min_size:
        frame.setMinimumSize(QtCore.QSize(*min_size))
    frame.setStyleSheet(style)
    return frame

def create_styled_label(parent, text, font_size=12, bold=False, style="color: #2c3e50; padding: 10px;"):
    """Create a styled QLabel with customizable font size and boldness."""
    label = QtWidgets.QLabel(parent)
    font = QtGui.QFont("Segoe UI", font_size)
    if bold:
        font.setBold(True)
        font.setWeight(75)
    label.setFont(font)
    label.setStyleSheet(style)
    label.setText(text)
    return label

def create_styled_button(parent, text, min_size=None):
    """Create a styled QPushButton with hover and pressed effects."""
    button = QtWidgets.QPushButton(parent)
    if min_size:
        button.setMinimumSize(QtCore.QSize(*min_size))
    button.setStyleSheet("""
        QPushButton {
            background-color: #3498db;
            color: white;
            font-family: 'Segoe UI';
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px;
            border: none;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #1c6ea4;
        }
    """)
    button.setText(text)
    return button

def create_input_field(parent, label_text, min_label_size=(120, 0)):
    """Create a horizontal layout with a label and a QLineEdit."""
    frame = create_styled_frame(parent, style="padding: 7px;")
    layout = QtWidgets.QHBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    
    label = create_styled_label(frame, label_text, font_size=12, bold=True, style="color: #2c3e50;")
    if min_label_size:
        label.setMinimumSize(QtCore.QSize(*min_label_size))
    
    line_edit = QtWidgets.QLineEdit(frame)
    line_edit.setStyleSheet("background-color: rgb(168, 168, 168);")
    
    layout.addWidget(label)
    layout.addWidget(line_edit)
    return frame, line_edit

def create_login_page(parent ,title, name_field_text="Name :", password_field_text="Password :", submit_text="Submit",):
    """Create a login page with a title, name and password fields, and a submit button."""
    page = QtWidgets.QWidget(parent)
    main_layout = QtWidgets.QVBoxLayout(page)
    
    # Header frame with title
    header_frame = create_styled_frame(page, style="background-color: #ffffff; border-radius: 10px; padding: 10px;")
    header_layout = QtWidgets.QVBoxLayout(header_frame)
    title_label = create_styled_label(header_frame, title, font_size=30)
    header_layout.addWidget(title_label, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
    main_layout.addWidget(header_frame, 0, QtCore.Qt.AlignTop)
    
    # Content frame
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    
    # Form frame
    form_frame = create_styled_frame(content_frame, min_size=(340, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(20)
    
    # Input fields
    name_frame, name_edit = create_input_field(form_frame, name_field_text)
    password_frame, password_edit = create_input_field(form_frame, password_field_text)
    
    # Submit button
    button_frame = create_styled_frame(form_frame, style="padding: 7px;")
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    button_layout.setSpacing(60)
    submit_button = create_styled_button(button_frame, submit_text, min_size=(150, 0))
    button_layout.addWidget(submit_button, 0, QtCore.Qt.AlignHCenter)
    
    form_layout.addWidget(name_frame)
    form_layout.addWidget(password_frame)
    form_layout.addWidget(button_frame)
    
    content_layout.addWidget(form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    submit_button.clicked.connect(lambda: on_login_button_clicked(parent,name_edit, password_edit))
    
    return page, name_edit, password_edit, submit_button
def on_login_button_clicked(parent,name_field, password_field):
    # Get the entered name and password
    name = name_field.text()
    password = password_field.text()
    # Check if the entered name and password are correct
    if name == "" and password == "":
        # Show a message box with the entered name and password
        Dialog = QtWidgets.QDialog()
        Dialog.setObjectName("Dialog")
        Dialog.resize(317, 60)
        verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        verticalLayout.setObjectName("verticalLayout")
        label = QtWidgets.QLabel(Dialog)
        label.setObjectName("label")
        label.setText("Please enter both name and password")
        verticalLayout.addWidget(label, 0, QtCore.Qt.AlignTop)
        buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        buttonBox.setOrientation(QtCore.Qt.Horizontal)
        buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        buttonBox.setObjectName("buttonBox")
        verticalLayout.addWidget(buttonBox)

        buttonBox.accepted.connect(Dialog.accept) # type: ignore
        buttonBox.rejected.connect(lambda:rejectBTN())# type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        def rejectBTN():
            parent.setCurrentIndex(0)
            Dialog.reject()
        # Show the dialog
        Dialog.exec_()
    else:
        print(f"Name: {name}, Password: {password}")
        
def create_home_page(parent, on_admin_clicked, on_employee_clicked, on_exit_clicked):
    """Create the home page with Admin, Employee, and Exit buttons."""
    page = QtWidgets.QWidget(parent)
    main_layout = QtWidgets.QVBoxLayout(page)
    main_layout.setContentsMargins(20, 20, 20, 20)
    main_layout.setSpacing(20)
    
    # Header frame with title
    header_frame = create_styled_frame(page, style="background-color: #ffffff; border-radius: 10px; padding: 10px;")
    header_layout = QtWidgets.QVBoxLayout(header_frame)
    title_label = create_styled_label(header_frame, "Bank Management System", font_size=30)
    header_layout.addWidget(title_label, 0, QtCore.Qt.AlignHCenter)
    main_layout.addWidget(header_frame, 0, QtCore.Qt.AlignTop)
    
    # Button frame
    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    
    # Button container
    button_container = create_styled_frame(button_frame, min_size=(300, 0), style="background-color: #ffffff; border-radius: 15px; padding: 20px;")
    button_container_layout = QtWidgets.QVBoxLayout(button_container)
    button_container_layout.setSpacing(15)
    
    # Buttons
    admin_button = create_styled_button(button_container, "Admin")
    employee_button = create_styled_button(button_container, "Employee")
    exit_button = create_styled_button(button_container, "Exit")
    exit_button.setStyleSheet("""
        QPushButton {
            background-color: #e74c3c;
            color: white;
            font-family: 'Segoe UI';
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px;
            border: none;
        }
        QPushButton:hover {
            background-color: #c0392b;
        }
        QPushButton:pressed {
            background-color: #992d22;
        }
    """)
    
    button_container_layout.addWidget(admin_button)
    button_container_layout.addWidget(employee_button)
    button_container_layout.addWidget(exit_button)
    
    button_layout.addWidget(button_container, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(button_frame)
    
    # Connect button signals
    admin_button.clicked.connect(on_admin_clicked)
    employee_button.clicked.connect(on_employee_clicked)
    exit_button.clicked.connect(on_exit_clicked)
    
    return page

def create_admin_menu_page(perent):
    """Create the admin menu page with buttons for adding, deleting, and viewing accounts."""
    page = QtWidgets.QWidget(perent)
    main_layout = QtWidgets.QVBoxLayout(page)
    main_layout.setContentsMargins(20, 20, 20, 20)
    main_layout.setSpacing(20)

    # Header frame with title
    header_frame = create_styled_frame(page, style="background-color: #ffffff; border-radius: 10px; padding: 10px;")
    header_layout = QtWidgets.QVBoxLayout(header_frame)
    title_label = create_styled_label(header_frame, "Admin Menu", font_size=30)
    header_layout.addWidget(title_label, 0, QtCore.Qt.AlignHCenter)
    main_layout.addWidget(header_frame, 0, QtCore.Qt.AlignTop)

    # Button frame
    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    # Button container
    button_container = create_styled_frame(button_frame, min_size=(300, 0), style="background-color: #ffffff; border-radius: 15px; padding: 20px;")
    button_container_layout = QtWidgets.QVBoxLayout(button_container)
    button_container_layout.setSpacing(15)
    # Buttons
    add_button = create_styled_button(button_container, "Add Employee")
    update_employee = create_styled_button(button_container, "Update Employee")
    employee_list = create_styled_button(button_container, "Employee List")
    total_money = create_styled_button(button_container, "Total Money")
    back_to_home = create_styled_button(button_container, "Back")
    button_container_layout.addWidget(add_button)
    button_container_layout.addWidget(update_employee)
    button_container_layout.addWidget(employee_list)
    button_container_layout.addWidget(total_money)
    button_container_layout.addWidget(back_to_home)
    button_layout.addWidget(button_container, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(button_frame)
    # Connect button signals
    # add_button.clicked.connect(on_add_employee_clicked)
    # update_employee.clicked.connect(on_update_employee_clicked)
    # employee_list.clicked.connect(on_employee_list_clicked)
    # total_money.clicked.connect(on_total_money_clicked)
    # back_to_home.clicked.connect(on_back_to_home_clicked)
    return page
    

def setup_main_window(main_window):
    """Set up the main window with a stacked widget containing home, admin, and employee pages."""
    main_window.setObjectName("MainWindow")
    main_window.resize(800, 600)
    main_window.setStyleSheet("background-color: #f0f2f5;")
    
    central_widget = QtWidgets.QWidget(main_window)
    main_layout = QtWidgets.QHBoxLayout(central_widget)
    
    stacked_widget = QtWidgets.QStackedWidget(central_widget)
    
    # Create pages
    def switch_to_admin():
        stacked_widget.setCurrentIndex(1)
    
    def switch_to_employee():
        stacked_widget.setCurrentIndex(2)
    
    def exit_app():
        QtWidgets.QApplication.quit()
        
    def admin_login_menu_page(name, password):
        result = backend.check_admin(name, password)
        if result:
            stacked_widget.setCurrentIndex(3) 
        else:
            print("Invalid admin credentials")  
        
    home_page = create_home_page(stacked_widget, switch_to_admin, switch_to_employee, exit_app)
    admin_page, admin_name, admin_password, admin_submit = create_login_page(stacked_widget, "Admin Login")
    admin_submit.clicked.connect(
        lambda: admin_login_menu_page(admin_name.text(), admin_password.text())
    )
    admin_menu_page = create_admin_menu_page(stacked_widget)
    
    employee_page, employee_name, employee_password, employee_submit = create_login_page(stacked_widget, "Employee Login")
    
    
    # Add pages to stacked widget
    stacked_widget.addWidget(home_page)
    stacked_widget.addWidget(admin_page)
    stacked_widget.addWidget(employee_page)
    stacked_widget.addWidget(admin_menu_page)
    
    main_layout.addWidget(stacked_widget)
    main_window.setCentralWidget(central_widget)
    
    # Set initial page
    stacked_widget.setCurrentIndex(0)
    
    return stacked_widget, {
        "admin_name": admin_name,
        "admin_password": admin_password,
        "admin_submit": admin_submit,
        "employee_name": employee_name,
        "employee_password": employee_password,
        "employee_submit": employee_submit
    }

def main():
    """Main function to launch the application."""
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    stacked_widget, widgets = setup_main_window(main_window)
    
    # Example: Connect submit buttons to print input values

    
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

