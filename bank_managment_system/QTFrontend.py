
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
def show_popup_message(parent, message: str, page: int = None, show_cancel: bool = True):
    """Reusable popup message box.

    Args:
        parent: The parent widget.
        message (str): The message to display.
        page (int, optional): Page index to switch to after dialog closes.
        show_cancel (bool): Whether to show the Cancel button.
    """
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Message")
    dialog.setFixedSize(350, 100)
    dialog.setStyleSheet("background-color: #f0f0f0;")
    
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setSpacing(10)
    layout.setContentsMargins(15, 15, 15, 15)
    
    label = QtWidgets.QLabel(message)
    label.setStyleSheet("font-size: 12px; color: #2c3e50;")
    label.setWordWrap(True)
    layout.addWidget(label)
    
    # Decide which buttons to show
    if show_cancel:
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
    else:
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
    
    button_box.setStyleSheet("""
        QPushButton {
            background-color: #3498db;
            color: white;
            border-radius: 4px;
            padding: 6px 12px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #1c6ea4;
        }
    """)
    layout.addWidget(button_box)
    
    # Connect buttons
    def on_accept():
        if page is not None:
            parent.setCurrentIndex(page)
        dialog.accept()
    
    def on_reject():
        if page is not None:
            parent.setCurrentIndex(page)
        dialog.reject()
    
    button_box.accepted.connect(on_accept)
    button_box.rejected.connect(on_reject)
    
    dialog.exec_()
def get_employee_name(parent, name_field_text="Enter Employee Name"):
    page, main_layout = create_page_with_header(parent, "Employee Data Update")
    
    # Content frame
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    
    # Form frame
    form_frame = create_styled_frame(content_frame, min_size=(340, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    
    # Form fields
    name_label, name_field = create_input_field(form_frame, name_field_text)
    search_button = create_styled_button(form_frame, "Search", min_size=(100, 30))
    form_layout.addWidget(name_label)
    form_layout.addWidget(search_button)
    content_layout.addWidget(form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    def on_search_button_clicked():
        entered_name = name_field.text().strip()
        if not entered_name:
            QtWidgets.QMessageBox.warning(parent, "Input Error", "Please enter an employee name.")
            return
    
        try:
            cur = backend.cur
            cur.execute("SELECT * FROM staff WHERE name = ?", (entered_name,))
            fetch = cur.fetchone()
            if fetch:
                QtWidgets.QMessageBox.information(parent, "Employee Found",
                                                  f"Employee data:\nID: {fetch[0]}\nName: {fetch[1]}\nDept: {fetch[2]}\nRole: {fetch[3]}")
            else:
                QtWidgets.QMessageBox.information(parent, "Not Found", "Employee not found.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(parent, "Error", f"An error occurred: {str(e)}")
    
    search_button.clicked.connect(on_search_button_clicked)
    
    return page

            
        #backend.check_name_in_staff()
def create_login_page(parent ,title, name_field_text="Name :", password_field_text="Password :", submit_text="Submit",):
    """Create a login page with a title, name and password fields, and a submit button."""
    page, main_layout = create_page_with_header(parent, "Admin Menu")
    
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
    
    
    
    return page, name_edit, password_edit, submit_button
def on_login_button_clicked(parent, name_field, password_field):
    name = name_field.text().strip()
    password = password_field.text().strip()
    
    if not name or not password:
        show_popup_message(parent, "Please enter your name and password.", 0)
    else:
        try:
            # Ideally, here you'd call a backend authentication check
            success = backend.check_admin(name, password)
            if success:
                QtWidgets.QMessageBox.information(parent, "Login Successful", f"Welcome, {name}!")
            else:
                QtWidgets.QMessageBox.warning(parent, "Login Failed", "Incorrect name or password.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(parent, "Error", f"An error occurred during login: {str(e)}")
        
def create_home_page(parent, on_admin_clicked, on_employee_clicked, on_exit_clicked):
    """Create the home page with Admin, Employee, and Exit buttons."""
    page, main_layout = create_page_with_header(parent, "Admin Menu")
    
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
def create_page_with_header(parent, title_text):
    """Create a page with a styled header and return the page + main layout."""
    page = QtWidgets.QWidget(parent)
    main_layout = QtWidgets.QVBoxLayout(page)
    main_layout.setContentsMargins(20, 20, 20, 20)
    main_layout.setSpacing(20)

    header_frame = create_styled_frame(page, style="background-color: #ffffff; border-radius: 10px; padding: 10px;")
    header_layout = QtWidgets.QVBoxLayout(header_frame)
    title_label = create_styled_label(header_frame, title_text, font_size=30)
    header_layout.addWidget(title_label, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
    
    main_layout.addWidget(header_frame, 0, QtCore.Qt.AlignTop)
    return page, main_layout
def create_admin_menu_page(parent):
    page, main_layout = create_page_with_header(parent, "Admin Menu")

    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    button_layout = QtWidgets.QVBoxLayout(button_frame)

    button_container = create_styled_frame(button_frame, min_size=(300, 0), style="background-color: #ffffff; border-radius: 15px; padding: 20px;")
    button_container_layout = QtWidgets.QVBoxLayout(button_container)
    button_container_layout.setSpacing(15)

    # Define button labels
    button_labels = ["Add Employee", "Update Employee", "Employee List", "Total Money", "Back"]
    buttons = []

    for label in button_labels:
        btn = create_styled_button(button_container, label)
        button_container_layout.addWidget(btn)
        buttons.append(btn)

    button_layout.addWidget(button_container, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(button_frame)

    return page, *buttons  # Unpack as add_button, update_employee, etc.

    
def create_add_employee_page(parent, title, submit_text="Submit",update_btn:bool=False):
    page, main_layout = create_page_with_header(parent, title)

    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    form_frame = create_styled_frame(content_frame, min_size=(340, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(20)

    # Define input fields
    fields = ["Name :", "Password :", "Salary :", "Position :"]
    edits = []

    for field in fields:
        field_frame, field_edit = create_input_field(form_frame, field)
        form_layout.addWidget(field_frame)
        edits.append(field_edit)

    # Submit button
    button_frame = create_styled_frame(form_frame, style="padding: 7px;")
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    if update_btn:
        update_button = create_styled_button(button_frame, "Update", min_size=(150, 0))
        button_layout.addWidget(update_button, 0, QtCore.Qt.AlignHCenter)
    else:
        submit_button = create_styled_button(button_frame, submit_text, min_size=(150, 0))
        button_layout.addWidget(submit_button, 0, QtCore.Qt.AlignHCenter)
    

    form_layout.addWidget(button_frame)
    content_layout.addWidget(form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    if update_btn:
        return page, *edits, update_button
    else:
        return page, *edits, submit_button  # Unpack as name_edit, password_edit, etc.
  
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
            try:
                # Ideally, here you'd call a backend authentication check
                success = backend.check_admin(name, password)
                if success:
                    QtWidgets.QMessageBox.information(stacked_widget, "Login Successful", f"Welcome, {name}!")
                    stacked_widget.setCurrentIndex(3)
                else:
                    QtWidgets.QMessageBox.warning(stacked_widget, "Login Failed", "Incorrect name or password.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(stacked_widget, "Error", f"An error occurred during login: {str(e)}")
                # show_popup_message(stacked_widget,"Invalid admin credentials",0)
            
    def add_employee_form_submit(name, password, salary, position):
        if (
            len(name) != 0
            and len(password) != 0
            and len(salary) != 0
            and len(position) != 0
        ):
            backend.create_employee(name, password, salary, position)
            show_popup_message(stacked_widget,"Employee added successfully",3,False)
            
        else:
            print("Please fill in all fields")
            show_popup_message(stacked_widget,"Please fill in all fields",3)
    def update_employee_data(name, password, salary, position, name_to_update):
        try:
            cur = backend.cur
            if name_to_update:
                cur.execute("UPDATE staff SET Name = ? WHERE name = ?", (name, name_to_update))
            
            cur.execute("UPDATE staff SET Name = ? WHERE name = ?", (password, name))
            cur.execute("UPDATE staff SET password = ? WHERE name = ?", (password, name))
            cur.execute("UPDATE staff SET salary = ? WHERE name = ?", (salary, name))
            cur.execute("UPDATE staff SET position = ? WHERE name = ?", (position, name))
            backend.conn.commit()
            show_popup_message(stacked_widget,"Employee Upadate successfully",3,False)
            
        except:
            show_popup_message(stacked_widget,"Please fill in all fields",3)

    def fetch_employee_data(name):
        try:
            cur = backend.cur
            cur.execute("SELECT * FROM staff WHERE name = ?", (name,))
            employee_data = cur.fetchone()
            return employee_data
        except:
            print("Error fetching employee data")
            return None
          
        
   # Create Home Page
    home_page = create_home_page(
        stacked_widget,
        switch_to_admin,
        switch_to_employee,
        exit_app
    )

    # Create Admin Login Page
    admin_page, admin_name, admin_password, admin_submit = create_login_page(
        stacked_widget,
        title="Admin Login"
    )
    admin_password.setEchoMode(QtWidgets.QLineEdit.Password)
    admin_name.setFont(QtGui.QFont("Arial", 10))
    admin_password.setFont(QtGui.QFont("Arial", 10))
    admin_name.setPlaceholderText("Enter your name")
    admin_password.setPlaceholderText("Enter your password")

    admin_submit.clicked.connect(
        lambda: admin_login_menu_page(
            admin_name.text(),
            admin_password.text()
        )
    )

    # Create Admin Menu Page
    admin_menu_page, add_button, update_button, list_button, money_button, back_button = create_admin_menu_page(
        stacked_widget
    )

    add_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(4))
    update_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(5))

    # Create Add Employee Page
    add_employee_page, emp_name, emp_password, emp_salary, emp_position, emp_submit = create_add_employee_page(
        stacked_widget,
        title="Add Employee"
    )
    
    # Update Employee Page
    update_employee_page1 = get_employee_name(stacked_widget)
    # apply the update_employee_data function to the submit button
    



    # ///////////////////////////
    emp_submit.clicked.connect(
        lambda: add_employee_form_submit(
            emp_name.text(),
            emp_password.text(),
            emp_salary.text(),
            emp_position.text()
        )
    )

    # Create Employee Login Page
    employee_page, employee_name, employee_password, employee_submit = create_login_page(
        stacked_widget,
        title="Employee Login"
    )
    
    
    # Add pages to stacked widget
    stacked_widget.addWidget(home_page)#0
    stacked_widget.addWidget(admin_page)#1
    stacked_widget.addWidget(employee_page)#2
    stacked_widget.addWidget(admin_menu_page)#3
    stacked_widget.addWidget(add_employee_page)#4
    stacked_widget.addWidget(update_employee_page1)#5
    
    main_layout.addWidget(stacked_widget)
    main_window.setCentralWidget(central_widget)
    
    # Set initial page
    stacked_widget.setCurrentIndex(5)
    
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

