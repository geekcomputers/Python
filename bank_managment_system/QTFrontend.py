import sys
from collections.abc import Callable
from typing import Any

import backendModule
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    if hasattr(backendModule, 'connect_database'):
        backendModule.connect_database()
    else:
        raise AttributeError("backend module missing 'connect_database' function")
except Exception as e:
    print(f"Database connection error: {e}")
    sys.exit(1)

employee_data: tuple[Any, ...] | None = None

HOME_PAGE = 0
ADMIN_PAGE = 1
EMPLOYEE_PAGE = 2
ADMIN_MENU_PAGE = 3
ADD_EMPLOYEE_PAGE = 4
UPDATE_EMPLOYEE_PAGE1 = 5
UPDATE_EMPLOYEE_PAGE2 = 6
EMPLOYEE_LIST_PAGE = 7
ADMIN_TOTAL_MONEY = 8
EMPLOYEE_MENU_PAGE = 9
EMPLOYEE_CREATE_ACCOUNT_PAGE = 10
EMPLOYEE_SHOW_DETAILS_PAGE1 = 11
EMPLOYEE_SHOW_DETAILS_PAGE2 = 12
EMPLOYEE_ADD_BALANCE_SEARCH = 13
EMPLOYEE_ADD_BALANCE_PAGE = 14
EMPLOYEE_WITHDRAW_MONEY_SEARCH = 15
EMPLOYEE_WITHDRAW_MONEY_PAGE = 16
EMPLOYEE_CHECK_BALANCE_SEARCH = 17
EMPLOYEE_CHECK_BALANCE_PAGE = 18
EMPLOYEE_UPDATE_ACCOUNT_SEARCH = 19
EMPLOYEE_UPDATE_ACCOUNT_PAGE = 20

FONT_SIZE = QtGui.QFont("Segoe UI", 12)

def create_styled_frame(parent: QtWidgets.QWidget, min_size: tuple[int, int] | None = None, style: str = "") -> QtWidgets.QFrame:
    """Create a styled QFrame with optional minimum size and custom style."""
    frame = QtWidgets.QFrame(parent)
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    frame.setFrameShadow(QtWidgets.QFrame.Raised)
    if min_size:
        frame.setMinimumSize(QtCore.QSize(*min_size))
    frame.setStyleSheet(style)
    return frame

def create_styled_label(parent: QtWidgets.QWidget, text: str, font_size: int = 12, bold: bool = False, style: str = "color: #2c3e50; padding: 10px;") -> QtWidgets.QLabel:
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

def create_styled_button(parent: QtWidgets.QWidget, text: str, min_size: tuple[int, int] | None = None) -> QtWidgets.QPushButton:
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

def create_input_field(parent: QtWidgets.QWidget, label_text: str, min_label_size: tuple[int, int] = (120, 0)) -> tuple[QtWidgets.QFrame, QtWidgets.QLineEdit]:
    """Create a horizontal layout with a label and input field."""
    frame = create_styled_frame(parent, style="padding: 7px;")
    layout = QtWidgets.QHBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    
    label = create_styled_label(frame, label_text, font_size=12, bold=True, style="color: #2c3e50;")
    if min_label_size:
        label.setMinimumSize(QtCore.QSize(*min_label_size))
    
    line_edit = QtWidgets.QLineEdit(frame)
    line_edit.setFont(FONT_SIZE)
    line_edit.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; padding: 8px;")
    
    layout.addWidget(label)
    layout.addWidget(line_edit)
    return frame, line_edit

def create_input_field_V(parent: QtWidgets.QWidget, label_text: str, min_label_size: tuple[int, int] = (120, 0)) -> tuple[QtWidgets.QFrame, QtWidgets.QLineEdit]:
    """Create a vertical layout with a label and input field."""
    frame = create_styled_frame(parent, style="padding: 7px;")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    
    label = create_styled_label(frame, label_text, font_size=12, bold=True, style="color: #2c3e50;")
    if min_label_size:
        label.setMinimumSize(QtCore.QSize(*min_label_size))
    
    line_edit = QtWidgets.QLineEdit(frame)
    line_edit.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; padding: 8px;")
    line_edit.setFont(FONT_SIZE)
    
    layout.addWidget(label)
    layout.addWidget(line_edit)
    return frame, line_edit

def show_popup_message(parent: QtWidgets.QWidget, message: str, page: int | None = None, show_cancel: bool = False, cancel_page: int = HOME_PAGE) -> None:
    """Display a reusable popup message box with optional navigation."""
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
    
    if show_cancel:
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
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
    
    def on_accept() -> None:
        if page is not None:
            parent.setCurrentIndex(page)
        dialog.accept()
    
    def on_reject() -> None:
        if page is not None:
            parent.setCurrentIndex(cancel_page)
        dialog.reject()
    
    button_box.accepted.connect(on_accept)
    button_box.rejected.connect(on_reject)
    
    dialog.exec_()

def search_result(parent: QtWidgets.QWidget, title: str, label_text: str) -> tuple[QtWidgets.QWidget, tuple[QtWidgets.QLineEdit, QtWidgets.QPushButton]]:
    """Create a search page with input field and submit button."""
    page, main_layout = create_page_with_header(parent, title)
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    content_layout.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    
    form_frame = create_styled_frame(content_frame, min_size=(400, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(3)
    
    user_frame, user_account_number = create_input_field(form_frame, label_text, min_label_size=(180, 0))
    form_layout.addWidget(user_frame)
    user_account_number.setFont(FONT_SIZE)
    
    submit_button = create_styled_button(form_frame, "Submit", min_size=(100, 50))
    form_layout.addWidget(submit_button)
    
    content_layout.addWidget(form_frame, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    return page, (user_account_number, submit_button)

def create_page_with_header(parent: QtWidgets.QWidget, title_text: str) -> tuple[QtWidgets.QWidget, QtWidgets.QLayout]:
    """Create a page with a styled header and return the page and main layout."""
    page = QtWidgets.QWidget(parent)
    main_layout = QtWidgets.QVBoxLayout(page)
    main_layout.setContentsMargins(20, 20, 20, 20)
    main_layout.setSpacing(20)

    header_frame = create_styled_frame(page, style="background-color: #ffffff; border-radius: 10px; padding: 10px;")
    header_layout = QtWidgets.QVBoxLayout(header_frame)
    title_label = create_styled_label(header_frame, title_text, font_size=30)
    header_layout.addWidget(title_label, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
    
    main_layout.addWidget(header_frame, alignment=QtCore.Qt.AlignTop)
    return page, main_layout

def get_employee_name(parent: QtWidgets.QWidget, name_field_text: str = "Enter Employee Name") -> QtWidgets.QWidget:
    """Create a page for searching employees by name."""
    page, main_layout = create_page_with_header(parent, "Employee Data Update")
    
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    
    form_frame = create_styled_frame(content_frame, min_size=(340, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    
    name_label, name_field = create_input_field(form_frame, name_field_text)
    search_button = create_styled_button(form_frame, "Search", min_size=(100, 30))
    form_layout.addWidget(name_label)
    form_layout.addWidget(search_button)
    
    content_layout.addWidget(form_frame, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    def on_search_button_clicked() -> None:
        global employee_data
        entered_name = name_field.text().strip()
        
        if not entered_name:
            QtWidgets.QMessageBox.warning(parent, "Input Error", "Please enter an employee name.")
            return
    
        try:
            if not hasattr(backendModule, 'check_name_in_staff'):
                raise AttributeError("backend missing 'check_name_in_staff' function")

            employee_exists = backendModule.check_name_in_staff(entered_name)
            if employee_exists:
                if not hasattr(backendModule, 'get_employee_data'):
                    raise AttributeError("backend missing 'get_employee_data' function")

                employee_data = backendModule.get_employee_data(entered_name)
                parent.setCurrentIndex(UPDATE_EMPLOYEE_PAGE2)
            else:
                QtWidgets.QMessageBox.information(parent, "Not Found", "Employee not found.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(parent, "Error", f"An error occurred: {str(e)}")
    
    search_button.clicked.connect(on_search_button_clicked)
    return page

def create_login_page(parent: QtWidgets.QWidget, title: str, name_field_text: str = "Name :", password_field_text: str = "Password :", submit_text: str = "Submit") -> tuple[QtWidgets.QWidget, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QPushButton]:
    """Create a login page with name, password fields and submit button."""
    page, main_layout = create_page_with_header(parent, title)
    
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    
    form_frame = create_styled_frame(content_frame, min_size=(340, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(20)
    
    name_frame, name_edit = create_input_field(form_frame, name_field_text)
    password_frame, password_edit = create_input_field(form_frame, password_field_text)
    password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
    
    button_frame = create_styled_frame(form_frame, style="padding: 7px;")
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    button_layout.setSpacing(60)
    submit_button = create_styled_button(button_frame, submit_text, min_size=(150, 0))
    button_layout.addWidget(submit_button, alignment=QtCore.Qt.AlignHCenter)
    
    form_layout.addWidget(name_frame)
    form_layout.addWidget(password_frame)
    form_layout.addWidget(button_frame)
    
    content_layout.addWidget(form_frame, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    return page, name_edit, password_edit, submit_button

def on_login_button_clicked(parent: QtWidgets.QWidget, name_field: QtWidgets.QLineEdit, password_field: QtWidgets.QLineEdit) -> None:
    """Handle login button click event."""
    name = name_field.text().strip()
    password = password_field.text().strip()
    
    if not name or not password:
        show_popup_message(parent, "Please enter your name and password.", HOME_PAGE)
        return

    try:
        if not hasattr(backendModule, 'check_admin'):
            raise AttributeError("backend missing 'check_admin' function")

        success = backendModule.check_admin(name, password)
        if success:
            QtWidgets.QMessageBox.information(parent, "Login Successful", f"Welcome, {name}!")
        else:
            QtWidgets.QMessageBox.warning(parent, "Login Failed", "Incorrect name or password.")
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"An error occurred during login: {str(e)}")

def create_home_page(parent: QtWidgets.QWidget, on_admin_clicked: Callable[[], None], on_employee_clicked: Callable[[], None], on_exit_clicked: Callable[[], None]) -> QtWidgets.QWidget:
    """Create home page with navigation buttons."""
    page, main_layout = create_page_with_header(parent, "Bank Management System")
    
    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    
    button_container = create_styled_frame(button_frame, min_size=(300, 0), style="background-color: #ffffff; border-radius: 15px; padding: 20px;")
    button_container_layout = QtWidgets.QVBoxLayout(button_container)
    button_container_layout.setSpacing(15)
    
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
    
    button_layout.addWidget(button_container, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(button_frame)
    
    admin_button.clicked.connect(on_admin_clicked)
    employee_button.clicked.connect(on_employee_clicked)
    exit_button.clicked.connect(on_exit_clicked)
    
    return page

def create_admin_menu_page(parent: QtWidgets.QWidget) -> tuple[QtWidgets.QWidget, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton]:
    """Create admin menu page with function buttons."""
    page, main_layout = create_page_with_header(parent, "Admin Menu")

    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    button_layout = QtWidgets.QVBoxLayout(button_frame)

    button_container = create_styled_frame(button_frame, min_size=(300, 0), style="background-color: #ffffff; border-radius: 15px; padding: 20px;")
    button_container_layout = QtWidgets.QVBoxLayout(button_container)
    button_container_layout.setSpacing(15)

    button_labels = ["Add Employee", "Update Employee", "Employee List", "Total Money", "Back"]
    buttons: list[QtWidgets.QPushButton] = []

    for label in button_labels:
        btn = create_styled_button(button_container, label)
        button_container_layout.addWidget(btn)
        buttons.append(btn)

    button_layout.addWidget(button_container, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(button_frame)

    return page, *buttons

def create_add_employee_page(parent: QtWidgets.QWidget, title: str, submit_text: str = "Submit", update_btn: bool = False) -> tuple[QtWidgets.QWidget, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QPushButton]:
    """Create page for adding or updating employee information."""
    page, main_layout = create_page_with_header(parent, title)

    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    form_frame = create_styled_frame(content_frame, min_size=(340, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(10)

    fields = ["Name :", "Password :", "Salary :", "Position :"]
    name_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
    password_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
    salary_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit()
    position_edit: QtWidgets.QLineEdit = QtWidgets.QLineEdit()

    for i, field in enumerate(fields):
        field_frame, field_edit = create_input_field(form_frame, field)
        form_layout.addWidget(field_frame)
        if i == 0:
            name_edit = field_edit
        elif i == 1:
            password_edit = field_edit
        elif i == 2:
            salary_edit = field_edit
        elif i == 3:
            position_edit = field_edit

    button_frame = create_styled_frame(form_frame, style="padding: 7px;")
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    
    if update_btn:
        submit_button = create_styled_button(button_frame, "Update", min_size=(100, 50))
    else:
        submit_button = create_styled_button(button_frame, submit_text, min_size=(100, 50))
    
    button_layout.addWidget(submit_button, alignment=QtCore.Qt.AlignHCenter)
    form_layout.addWidget(button_frame)
    
    content_layout.addWidget(form_frame, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    back_btn = QtWidgets.QPushButton("Back", content_frame)
    back_btn.setStyleSheet("""
        QPushButton {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
    """)
    back_btn.clicked.connect(lambda: parent.setCurrentIndex(ADMIN_MENU_PAGE))
    main_layout.addWidget(back_btn, alignment=QtCore.Qt.AlignLeft)
    
    return page, name_edit, password_edit, salary_edit, position_edit, submit_button

def show_employee_list_page(parent: QtWidgets.QWidget, title: str) -> QtWidgets.QWidget:
    """Create page to display list of employees."""
    page, main_layout = create_page_with_header(parent, title)
    
    content_frame = create_styled_frame(page, style="background-color: #f9f9f9; border-radius: 10px; padding: 15px;")
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    table_frame = create_styled_frame(content_frame, style="background-color: #ffffff; border-radius: 8px; padding: 10px;")
    table_layout = QtWidgets.QVBoxLayout(table_frame)
    table_layout.setSpacing(0)

    header_frame = create_styled_frame(table_frame, style="background-color: #f5f5f5; border-radius: 8px 8px 0 0; padding: 10px;")
    header_layout = QtWidgets.QHBoxLayout(header_frame)
    header_layout.setContentsMargins(10, 5, 10, 5)
    
    headers = ["Name", "Position", "Salary"]
    for i, header in enumerate(headers):
        header_label = QtWidgets.QLabel(header, header_frame)
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #333333;")
        if i == 2:
            header_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        else:
            header_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        header_layout.addWidget(header_label, 1 if i < 2 else 0)
    
    table_layout.addWidget(header_frame)

    try:
        if not hasattr(backendModule, 'show_employees_for_update'):
            raise AttributeError("backend missing 'show_employees_for_update' function")

        employees = backendModule.show_employees_for_update()
        for row, employee in enumerate(employees):
            row_frame = create_styled_frame(table_frame, style=f"background-color: {'#fafafa' if row % 2 else '#ffffff'}; padding: 8px;")
            row_layout = QtWidgets.QHBoxLayout(row_frame)
            row_layout.setContentsMargins(10, 5, 10, 5)

            name_label = QtWidgets.QLabel(str(employee[0]), row_frame)
            name_label.setStyleSheet("font-size: 14px; color: #333333;")
            name_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            row_layout.addWidget(name_label, 1)

            position_label = QtWidgets.QLabel(str(employee[3]), row_frame)
            position_label.setStyleSheet("font-size: 14px; color: #333333;")
            position_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            row_layout.addWidget(position_label, 1)

            salary_label = QtWidgets.QLabel(f"${float(employee[2]):,.2f}", row_frame)
            salary_label.setStyleSheet("font-size: 14px; color: #333333;")
            salary_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row_layout.addWidget(salary_label, 0)

            table_layout.addWidget(row_frame)
    except Exception as e:
        error_label = QtWidgets.QLabel(f"Error loading employees: {str(e)}", table_frame)
        table_layout.addWidget(error_label)

    table_layout.addStretch()
    
    back_button = QtWidgets.QPushButton("Back", content_frame)
    back_button.setStyleSheet("""
        QPushButton {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
    """)
    back_button.clicked.connect(lambda: parent.setCurrentIndex(ADMIN_MENU_PAGE))
    
    content_layout.addWidget(table_frame)
    main_layout.addWidget(back_button, alignment=QtCore.Qt.AlignLeft)
    main_layout.addWidget(content_frame)
    
    return page

def show_total_money(parent: QtWidgets.QWidget, title: str) -> QtWidgets.QWidget:
    """Create page to display total money."""
    page, main_layout = create_page_with_header(parent, title)

    content_frame = create_styled_frame(page, style="background-color: #f9f9f9; border-radius: 10px; padding: 15px;")
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    content_layout.setSpacing(10)

    try:
        if not hasattr(backendModule, 'all_money'):
            raise AttributeError("backend missing 'all_money' function")

        total_money = backendModule.all_money()
        total_money = int(total_money) if total_money is not None else 0
    except Exception as e:
        total_money = f"Error: {str(e)}"

    total_money_label = QtWidgets.QLabel(f"Total Money: ${total_money}", content_frame)
    total_money_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333333;")
    content_layout.addWidget(total_money_label, alignment=QtCore.Qt.AlignCenter)
    
    back_button = QtWidgets.QPushButton("Back", content_frame)
    back_button.setStyleSheet("""
        QPushButton {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
    """)
    back_button.clicked.connect(lambda: parent.setCurrentIndex(ADMIN_MENU_PAGE))
    
    content_layout.addWidget(back_button, alignment=QtCore.Qt.AlignCenter)
    main_layout.addWidget(content_frame)
    
    return page

def create_employee_menu_page(parent: QtWidgets.QWidget, title: str) -> tuple[QtWidgets.QWidget, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton, QtWidgets.QPushButton]:
    """Create employee menu page with function buttons."""
    page, main_layout = create_page_with_header(parent, title)

    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    button_layout = QtWidgets.QVBoxLayout(button_frame)

    button_container = create_styled_frame(button_frame, min_size=(300, 0), style="background-color: #ffffff; border-radius: 15px; padding: 20px;")
    button_container_layout = QtWidgets.QVBoxLayout(button_container)
    button_container_layout.setSpacing(15)

    button_labels = ["Create Account", "Show Details", "Add Balance", "Withdraw Money", "Check Balance", "Update Account", "List of All Members", "Delete Account", "Back"]
    buttons: list[QtWidgets.QPushButton] = []

    for label in button_labels:
        btn = create_styled_button(button_container, label)
        button_container_layout.addWidget(btn)
        buttons.append(btn)

    button_layout.addWidget(button_container, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(button_frame)

    return page, *buttons

def create_account_page(parent: QtWidgets.QWidget, title: str, update_btn: bool = False) -> tuple[QtWidgets.QWidget, tuple[QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QComboBox, QtWidgets.QPushButton]]:
    """Create page for creating or updating customer accounts."""
    page, main_layout = create_page_with_header(parent, title)

    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    form_frame = create_styled_frame(content_frame, min_size=(400, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(3)

    fields = ["Name :", "Age :", "Address", "Balance :", "Mobile number :"]
    edits: list[QtWidgets.QLineEdit] = []

    for field in fields:
        field_frame, field_edit = create_input_field(form_frame, field, min_label_size=(160, 0))
        form_layout.addWidget(field_frame)
        field_edit.setFont(QtGui.QFont("Arial", 12))
        edits.append(field_edit)

    account_type_label = QtWidgets.QLabel("Account Type :", form_frame)
    account_type_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333333;")
    form_layout.addWidget(account_type_label)
    
    account_type_dropdown = QtWidgets.QComboBox(form_frame)
    account_type_dropdown.addItems(["Savings", "Current", "Fixed Deposit"])
    account_type_dropdown.setStyleSheet("""
        QComboBox {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
            min-width: 200px;
            font-size: 14px;
        }
        QComboBox:hover {
            border: 1px solid #999;
        }
        QComboBox::drop-down {
            border: none;
            width: 25px;
        }
    """)
    form_layout.addWidget(account_type_dropdown)

    button_frame = create_styled_frame(form_frame, style="padding: 7px;")
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    
    if update_btn:
        submit_button = create_styled_button(button_frame, "Update", min_size=(100, 50))
    else:
        submit_button = create_styled_button(button_frame, "Submit", min_size=(100, 50))
    
    button_layout.addWidget(submit_button, alignment=QtCore.Qt.AlignHCenter)
    form_layout.addWidget(button_frame)
    
    content_layout.addWidget(form_frame, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    back_btn = QtWidgets.QPushButton("Back", content_frame)
    back_btn.setStyleSheet("""
        QPushButton {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
    """)
    back_btn.clicked.connect(lambda: parent.setCurrentIndex(EMPLOYEE_MENU_PAGE))
    main_layout.addWidget(back_btn, alignment=QtCore.Qt.AlignLeft)
    
    while len(edits) < 5:
        edits.append(QtWidgets.QLineEdit())
    
    return page, (edits[0], edits[1], edits[2], edits[3], edits[4], account_type_dropdown, submit_button)

def create_show_details_page1(parent: QtWidgets.QWidget, title: str) -> tuple[QtWidgets.QWidget, tuple[QtWidgets.QLineEdit, QtWidgets.QPushButton]]:
    """Create first page for showing account details (search by account number)."""
    page, main_layout = create_page_with_header(parent, title)
    
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    
    form_frame = create_styled_frame(content_frame, min_size=(400, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(3)
    
    user_frame, user_account_number = create_input_field(form_frame, "Enter Bank account Number :", min_label_size=(180, 0))
    form_layout.addWidget(user_frame)
    
    submit_button = create_styled_button(form_frame, "Submit", min_size=(100, 50))
    form_layout.addWidget(submit_button)
    
    content_layout.addWidget(form_frame, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    return page, (user_account_number, submit_button)

def create_show_details_page2(parent: QtWidgets.QWidget, title: str) -> tuple[QtWidgets.QWidget, tuple[QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QPushButton]]:
    """Create second page for showing account details (display details)."""
    page, main_layout = create_page_with_header(parent, title)
    
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    
    form_frame = create_styled_frame(content_frame, min_size=(400, 300), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    form_layout.setSpacing(3)
    
    labels = ["Account No: ", "Name: ", "Age:", "Address: ", "Balance: ", "Mobile Number: ", "Account Type: "]
    fields: list[QtWidgets.QLineEdit] = []
    
    for label in labels:
        label_frame, input_field = create_input_field(form_frame, label, min_label_size=(180, 30))
        form_layout.addWidget(label_frame)
        input_field.setReadOnly(True)
        input_field.setFont(QtGui.QFont("Arial", 12))
        fields.append(input_field)
    
    exit_btn = create_styled_button(form_frame, "Exit", min_size=(100, 50))
    exit_btn.setStyleSheet("""
        QPushButton {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
    """)
    exit_btn.clicked.connect(lambda: parent.setCurrentIndex(EMPLOYEE_MENU_PAGE))
    form_layout.addWidget(exit_btn)
    
    content_layout.addWidget(form_frame, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    main_layout.addWidget(content_frame)
    
    return page, (fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], exit_btn)

def update_user(parent: QtWidgets.QWidget, title: str, input_fields_label: str, input_field: bool = True) -> tuple[QtWidgets.QWidget, tuple[QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QPushButton] | tuple[QtWidgets.QLineEdit, QtWidgets.QLineEdit, QtWidgets.QPushButton]]:
    """Create page for updating user balance (add/withdraw)."""
    page, main_layout = create_page_with_header(parent, title)
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    content_layout.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    
    form_frame = create_styled_frame(content_frame, min_size=(400, 200), style="background-color: #ffffff; border-radius: 15px; padding: 10px;")
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(3)
    
    user_frame, user_account_name = create_input_field(form_frame, "User Name: ", min_label_size=(180, 0))
    balance_frame, user_balance = create_input_field(form_frame, "Balance: ", min_label_size=(180, 0))
    
    form_layout.addWidget(user_frame)
    form_layout.addWidget(balance_frame)
    
    user_account_name.setReadOnly(True)
    user_account_name.setStyleSheet("background-color: #8a8a8a; border: 1px solid #ccc; border-radius: 4px; padding: 8px;")
    user_balance.setReadOnly(True)
    user_balance.setStyleSheet("background-color: #8a8a8a; border: 1px solid #ccc; border-radius: 4px; padding: 8px;")
    
    amount_field: QtWidgets.QLineEdit | None = None
    if input_field:
        amount_frame, amount_field = create_input_field_V(form_frame, input_fields_label, min_label_size=(180, 0))
        form_layout.addWidget(amount_frame)
        if amount_field:
            amount_field.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; padding: 8px;")
    
    user_account_name.setFont(FONT_SIZE)
    user_balance.setFont(FONT_SIZE)
    if amount_field:
        amount_field.setFont(FONT_SIZE)
    
    submit_button = create_styled_button(form_frame, "Submit", min_size=(100, 50))
    form_layout.addWidget(submit_button)
    
    content_layout.addWidget(form_frame)
    main_layout.addWidget(content_frame)
    
    back_btn = create_styled_button(content_frame, "Back", min_size=(100, 50))
    back_btn.setStyleSheet("""
        QPushButton {
            background-color: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
    """)
    back_btn.clicked.connect(lambda: parent.setCurrentIndex(EMPLOYEE_MENU_PAGE))
    main_layout.addWidget(back_btn)
    
    if input_field and amount_field:
        return page, (user_account_name, user_balance, amount_field, submit_button)
    else:
        return page, (user_account_name, user_balance, submit_button)

def setup_main_window(main_window: QtWidgets.QMainWindow) -> tuple[QtWidgets.QStackedWidget, dict]:
    """Set up the main window with a stacked widget containing all pages."""
    main_window.setObjectName("MainWindow")
    main_window.resize(800, 600)
    main_window.setStyleSheet("background-color: #f0f2f5;")
    
    central_widget = QtWidgets.QWidget(main_window)
    main_layout = QtWidgets.QHBoxLayout(central_widget)
    
    stacked_widget = QtWidgets.QStackedWidget(central_widget)
    
    def switch_to_admin() -> None:
        stacked_widget.setCurrentIndex(ADMIN_PAGE)
    
    def switch_to_employee() -> None:
        stacked_widget.setCurrentIndex(EMPLOYEE_PAGE)
    
    def exit_app() -> None:
        QtWidgets.QApplication.quit()
    
    def admin_login_menu_page(name: str, password: str) -> None:
        try:
            if not hasattr(backendModule, 'check_admin'):
                raise AttributeError("backend missing 'check_admin' function")

            success = backendModule.check_admin(name, password)
            if success:
                QtWidgets.QMessageBox.information(stacked_widget, "Login Successful", f"Welcome, {name}!")
                stacked_widget.setCurrentIndex(ADMIN_MENU_PAGE)
            else:
                QtWidgets.QMessageBox.warning(stacked_widget, "Login Failed", "Incorrect name or password.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(stacked_widget, "Error", f"Login error: {str(e)}")
    
    def add_employee_form_submit(name: str, password: str, salary: str, position: str) -> None:
        if all([name, password, salary, position]):
            try:
                if not hasattr(backendModule, 'create_employee'):
                    raise AttributeError("backend missing 'create_employee' function")

                backendModule.create_employee(name, password, salary, position)
                show_popup_message(stacked_widget, "Employee added successfully", ADMIN_MENU_PAGE)
            except Exception as e:
                show_popup_message(stacked_widget, f"Error adding employee: {str(e)}", ADD_EMPLOYEE_PAGE)
        else:
            show_popup_message(stacked_widget, "Please fill in all fields", ADD_EMPLOYEE_PAGE)
    
    def handle_update_employee_data(name: str, password: str, salary: str, position: str, name_to_update: str) -> None:
        try:
            if not name_to_update:
                show_popup_message(stacked_widget, "Original employee name is missing.", UPDATE_EMPLOYEE_PAGE2)
                return
                
            if not any([name, password, salary, position]):
                show_popup_message(stacked_widget, "Please fill at least one field to update.", UPDATE_EMPLOYEE_PAGE2)
                return

            if name and hasattr(backendModule, 'update_employee_name'):
                backendModule.update_employee_name(name, name_to_update)
            if password and hasattr(backendModule, 'update_employee_password'):
                backendModule.update_employee_password(password, name_to_update)
            if salary and hasattr(backendModule, 'update_employee_salary'):
                try:
                    backendModule.update_employee_salary(float(salary), name_to_update)
                except ValueError:
                    show_popup_message(stacked_widget, "Salary must be a valid number.", UPDATE_EMPLOYEE_PAGE2)
                    return
            if position and hasattr(backendModule, 'update_employee_position'):
                backendModule.update_employee_position(position, name_to_update)
                
            show_popup_message(stacked_widget, "Employee updated successfully", ADMIN_MENU_PAGE)
        except Exception as e:
            show_popup_message(stacked_widget, f"Update error: {str(e)}", UPDATE_EMPLOYEE_PAGE2)
    
    home_page = create_home_page(stacked_widget, switch_to_admin, switch_to_employee, exit_app)
    
    admin_page, admin_name, admin_password, admin_submit = create_login_page(stacked_widget, "Admin Login")
    admin_password.setEchoMode(QtWidgets.QLineEdit.Password)
    admin_name.setPlaceholderText("Enter your name")
    admin_password.setPlaceholderText("Enter your password")
    admin_submit.clicked.connect(lambda: admin_login_menu_page(admin_name.text(), admin_password.text()))
    
    admin_menu_page, add_button, update_button, list_button, money_button, back_button = create_admin_menu_page(stacked_widget)
    add_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(ADD_EMPLOYEE_PAGE))
    update_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(UPDATE_EMPLOYEE_PAGE1))
    list_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_LIST_PAGE))
    back_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(HOME_PAGE))
    money_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(ADMIN_TOTAL_MONEY))
    
    add_employee_page, emp_name, emp_password, emp_salary, emp_position, emp_submit = create_add_employee_page(stacked_widget, "Add Employee")
    emp_submit.clicked.connect(lambda: add_employee_form_submit(emp_name.text().strip(), emp_password.text().strip(), emp_salary.text().strip(), emp_position.text().strip()))
    
    u_employee_page1 = get_employee_name(stacked_widget)
    
    u_employee_page2, u_employee_name, u_employee_password, u_employee_salary, u_employee_position, u_employee_update = create_add_employee_page(stacked_widget, "Update Employee Details", update_btn=True)
    
    def populate_employee_data() -> None:
        global employee_data
        if employee_data:
            u_employee_name.setText(str(employee_data[0]))
            u_employee_password.setText(str(employee_data[1]))
            u_employee_salary.setText(str(employee_data[2]))
            u_employee_position.setText(str(employee_data[3]))
        else:
            for field in [u_employee_name, u_employee_password, u_employee_salary, u_employee_position]:
                field.clear()
    
    stacked_widget.currentChanged.connect(lambda index: populate_employee_data() if index == UPDATE_EMPLOYEE_PAGE2 else None)
    u_employee_update.clicked.connect(lambda: handle_update_employee_data(u_employee_name.text().strip(), u_employee_password.text().strip(), u_employee_salary.text().strip(), u_employee_position.text().strip(), employee_data[0] if employee_data else ""))
    
    employee_list_page = show_employee_list_page(stacked_widget, "Employee List")
    admin_total_money_page = show_total_money(stacked_widget, "Total Money")
    
    employee_page, employee_name, employee_password, employee_submit = create_login_page(stacked_widget, "Employee Login")
    employee_submit.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_MENU_PAGE))
    
    employee_menu_page, e_create_account, e_show_details, e_add_balance, e_withdraw_money, e_check_balance, e_update_account, e_list_members, e_delete_account, e_back = create_employee_menu_page(stacked_widget, "Employee Menu")
    
    e_create_account.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_CREATE_ACCOUNT_PAGE))
    e_show_details.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_SHOW_DETAILS_PAGE1))
    e_add_balance.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_ADD_BALANCE_SEARCH))
    e_withdraw_money.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_WITHDRAW_MONEY_SEARCH))
    e_check_balance.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_CHECK_BALANCE_SEARCH))
    e_update_account.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_UPDATE_ACCOUNT_SEARCH))
    e_back.clicked.connect(lambda: stacked_widget.setCurrentIndex(HOME_PAGE))
    
    employee_create_account_page, create_account_fields = create_account_page(stacked_widget, "Create Account")
    name_edit, age_edit, address_edit, balance_edit, mobile_edit, account_type, create_submit = create_account_fields
    
    def handle_create_account() -> None:
        try:
            name = name_edit.text().strip()
            age = age_edit.text().strip()
            address = address_edit.text().strip()
            balance = balance_edit.text().strip()
            mobile = mobile_edit.text().strip()
            acc_type = account_type.currentText()
            
            if not all([name, age, address, balance, mobile, acc_type]):
                show_popup_message(stacked_widget, "Please fill all fields", EMPLOYEE_CREATE_ACCOUNT_PAGE)
                return
                
            if not age.isdigit() or int(age) < 18:
                show_popup_message(stacked_widget, "Age must be a number and at least 18", EMPLOYEE_CREATE_ACCOUNT_PAGE)
                return
                
            if not balance.isdigit() or int(balance) < 0:
                show_popup_message(stacked_widget, "Balance must be a positive number", EMPLOYEE_CREATE_ACCOUNT_PAGE)
                return
                
            if not mobile.isdigit() or len(mobile) != 10:
                show_popup_message(stacked_widget, "Mobile number must be 10 digits", EMPLOYEE_CREATE_ACCOUNT_PAGE)
                return

            if not hasattr(backendModule, 'create_customer'):
                raise AttributeError("backend missing 'create_customer' function")

            backendModule.create_customer(name, age, address, balance, acc_type, mobile)

            for field in [name_edit, age_edit, address_edit, balance_edit, mobile_edit]:
                field.clear()
                
            show_popup_message(stacked_widget, "Account created successfully", EMPLOYEE_MENU_PAGE)
        except Exception as e:
            show_popup_message(stacked_widget, f"Error creating account: {str(e)}", EMPLOYEE_CREATE_ACCOUNT_PAGE)
    
    create_submit.clicked.connect(handle_create_account)
    
    show_details_page1, show_details_widgets1 = create_show_details_page1(stacked_widget, "Show Details")
    show_details_input1, show_details_btn1 = show_details_widgets1
    
    show_details_page2, show_details_widgets2 = create_show_details_page2(stacked_widget, "Account Details")
    (acc_no_field, name_field, age_field, address_field, balance_field, mobile_field, acc_type_field, exit_btn) = show_details_widgets2
    
    def handle_show_details_submit() -> None:
        try:
            account_number = int(show_details_input1.text().strip())
            if not hasattr(backendModule, 'check_acc_no'):
                raise AttributeError("backend missing 'check_acc_no' function")

            if backendModule.check_acc_no(account_number):
                if not hasattr(backendModule, 'get_details'):
                    raise AttributeError("backend missing 'get_details' function")

                account_data = backendModule.get_details(account_number)
                acc_no_field.setText(str(account_data[0]))
                name_field.setText(str(account_data[1]))
                age_field.setText(str(account_data[2]))
                address_field.setText(str(account_data[3]))
                balance_field.setText(str(account_data[4]))
                mobile_field.setText(str(account_data[5]))
                acc_type_field.setText(str(account_data[6]))
                stacked_widget.setCurrentIndex(EMPLOYEE_SHOW_DETAILS_PAGE2)
            else:
                show_popup_message(stacked_widget, "Account not found", EMPLOYEE_SHOW_DETAILS_PAGE1)
        except ValueError:
            show_popup_message(stacked_widget, "Please enter a valid account number", EMPLOYEE_SHOW_DETAILS_PAGE1)
        except Exception as e:
            show_popup_message(stacked_widget, f"Error: {str(e)}", EMPLOYEE_SHOW_DETAILS_PAGE1)
    
    show_details_btn1.clicked.connect(handle_show_details_submit)
    
    def setup_balance_operation(title_search: str, placeholder: str, title_form: str, action_text: str, success_msg: str, backend_func: str, search_idx: int, page_idx: int, need_input: bool = True) -> tuple[QtWidgets.QWidget, QtWidgets.QWidget]:
        search_page, search_widgets = search_result(stacked_widget, title_search, placeholder)
        search_input, search_btn = search_widgets
        
        form_page, form_widgets = update_user(stacked_widget, title_form, action_text, need_input)
        
        if need_input and len(form_widgets) == 4:
            name_field, balance_field, amount_field, action_btn = form_widgets
        else:
            name_field, balance_field, action_btn = form_widgets
            amount_field = None
        
        def on_search() -> None:
            try:
                account_number = int(search_input.text().strip())
                if not hasattr(backendModule, 'check_acc_no'):
                    raise AttributeError("backend missing 'check_acc_no' function")

                if backendModule.check_acc_no(account_number):
                    if not hasattr(backendModule, 'get_details'):
                        raise AttributeError("backend missing 'get_details' function")

                    account_data = backendModule.get_details(account_number)
                    name_field.setText(str(account_data[1]))
                    balance_field.setText(str(account_data[4]))
                    stacked_widget.setCurrentIndex(page_idx)
                else:
                    show_popup_message(stacked_widget, "Account not found", search_idx)
            except ValueError:
                show_popup_message(stacked_widget, "Please enter a valid account number", search_idx)
            except Exception as e:
                show_popup_message(stacked_widget, f"Search error: {str(e)}", search_idx)
        
        def on_action() -> None:
            try:
                account_number = int(search_input.text().strip())
                if not hasattr(backendModule, backend_func):
                    raise AttributeError(f"backend missing '{backend_func}' function")
                    
                if need_input and amount_field:
                    amount = float(amount_field.text().strip())
                    getattr(backendModule, backend_func)(amount, account_number)
                else:
                    getattr(backendModule, backend_func)(account_number)

                search_input.clear()
                name_field.clear()
                balance_field.clear()
                if amount_field:
                    amount_field.clear()
                    
                show_popup_message(stacked_widget, success_msg, EMPLOYEE_MENU_PAGE)
            except ValueError:
                show_popup_message(stacked_widget, "Please enter a valid amount", page_idx)
            except Exception as e:
                show_popup_message(stacked_widget, f"Operation error: {str(e)}", page_idx)
        
        search_btn.clicked.connect(on_search)
        action_btn.clicked.connect(on_action)
        
        return search_page, form_page
    
    add_balance_search_page, add_balance_page = setup_balance_operation("Add Balance", "Enter Account Number: ", "Add Balance", "Enter Amount: ", "Balance updated successfully", "update_balance", EMPLOYEE_ADD_BALANCE_SEARCH, EMPLOYEE_ADD_BALANCE_PAGE)
    
    withdraw_search_page, withdraw_page = setup_balance_operation("Withdraw Money", "Enter Account Number: ", "Withdraw Money", "Withdraw Amount: ", "Amount withdrawn successfully", "deduct_balance", EMPLOYEE_WITHDRAW_MONEY_SEARCH, EMPLOYEE_WITHDRAW_MONEY_PAGE)
    
    check_balance_search_page, check_balance_page = setup_balance_operation("Check Balance", "Enter Account Number: ", "Check Balance", "", "Balance checked successfully", "check_balance", EMPLOYEE_CHECK_BALANCE_SEARCH, EMPLOYEE_CHECK_BALANCE_PAGE, False)
    
    update_account_search_page, update_account_search_widgets = search_result(stacked_widget, "Update Account", "Enter Account Number: ")
    update_account_input, update_account_btn = update_account_search_widgets
    
    update_account_page, update_account_fields = create_account_page(stacked_widget, "Update Account", update_btn=True)
    (u_name_edit, u_age_edit, u_address_edit, u_balance_edit, u_mobile_edit, u_acc_type, u_submit_btn) = update_account_fields
    u_balance_edit.setReadOnly(True)
    
    def handle_update_search() -> None:
        try:
            account_number = int(update_account_input.text().strip())
            if not hasattr(backendModule, 'get_details'):
                raise AttributeError("backend missing 'get_details' function")

            account_data = backendModule.get_details(account_number)
            if account_data:
                u_name_edit.setText(str(account_data[1]))
                u_age_edit.setText(str(account_data[2]))
                u_address_edit.setText(str(account_data[3]))
                u_balance_edit.setText(str(account_data[4]))
                u_mobile_edit.setText(str(account_data[5]))
                u_acc_type.setCurrentText(str(account_data[6]))
                stacked_widget.setCurrentIndex(EMPLOYEE_UPDATE_ACCOUNT_PAGE)
            else:
                show_popup_message(stacked_widget, "Account not found", EMPLOYEE_UPDATE_ACCOUNT_SEARCH)
        except ValueError:
            show_popup_message(stacked_widget, "Please enter a valid account number", EMPLOYEE_UPDATE_ACCOUNT_SEARCH)
        except Exception as e:
            show_popup_message(stacked_widget, f"Error: {str(e)}", EMPLOYEE_UPDATE_ACCOUNT_SEARCH)
    
    def handle_account_update() -> None:
        try:
            account_number = int(update_account_input.text().strip())
            name = u_name_edit.text().strip()
            age = u_age_edit.text().strip()
            address = u_address_edit.text().strip()
            mobile = u_mobile_edit.text().strip()
            acc_type = u_acc_type.currentText()
            
            if not all([name, age, address, mobile, acc_type]):
                show_popup_message(stacked_widget, "Please fill all fields", EMPLOYEE_UPDATE_ACCOUNT_PAGE)
                return
                
            if not age.isdigit() or int(age) < 18:
                show_popup_message(stacked_widget, "Age must be a number and at least 18", EMPLOYEE_UPDATE_ACCOUNT_PAGE)
                return
                
            if not mobile.isdigit() or len(mobile) != 10:
                show_popup_message(stacked_widget, "Mobile number must be 10 digits", EMPLOYEE_UPDATE_ACCOUNT_PAGE)
                return
            
            update_functions = [
                ("update_name_in_bank_table", name),
                ("update_age_in_bank_table", age),
                ("update_address_in_bank_table", address),
                ("update_mobile_number_in_bank_table", mobile),
                ("update_acc_type_in_bank_table", acc_type)
            ]
            
            for func_name, value in update_functions:
                if not hasattr(backendModule, func_name):
                    raise AttributeError(f"backend missing '{func_name}' function")
                getattr(backendModule, func_name)(value, account_number)

            update_account_input.clear()
            show_popup_message(stacked_widget, "Account updated successfully", EMPLOYEE_MENU_PAGE)
        except ValueError:
            show_popup_message(stacked_widget, "Invalid account number", EMPLOYEE_UPDATE_ACCOUNT_PAGE)
        except Exception as e:
            show_popup_message(stacked_widget, f"Update error: {str(e)}", EMPLOYEE_UPDATE_ACCOUNT_PAGE)
    
    update_account_btn.clicked.connect(handle_update_search)
    u_submit_btn.clicked.connect(handle_account_update)
    
    stacked_widget.addWidget(home_page)
    stacked_widget.addWidget(admin_page)
    stacked_widget.addWidget(employee_page)
    stacked_widget.addWidget(admin_menu_page)
    stacked_widget.addWidget(add_employee_page)
    stacked_widget.addWidget(u_employee_page1)
    stacked_widget.addWidget(u_employee_page2)
    stacked_widget.addWidget(employee_list_page)
    stacked_widget.addWidget(admin_total_money_page)
    stacked_widget.addWidget(employee_menu_page)
    stacked_widget.addWidget(employee_create_account_page)
    stacked_widget.addWidget(show_details_page1)
    stacked_widget.addWidget(show_details_page2)
    stacked_widget.addWidget(add_balance_search_page)
    stacked_widget.addWidget(add_balance_page)
    stacked_widget.addWidget(withdraw_search_page)
    stacked_widget.addWidget(withdraw_page)
    stacked_widget.addWidget(check_balance_search_page)
    stacked_widget.addWidget(check_balance_page)
    stacked_widget.addWidget(update_account_search_page)
    stacked_widget.addWidget(update_account_page)
    
    main_layout.addWidget(stacked_widget)
    main_window.setCentralWidget(central_widget)
    
    stacked_widget.setCurrentIndex(HOME_PAGE)
    
    return stacked_widget, {
        "admin_name": admin_name,
        "admin_password": admin_password,
        "admin_submit": admin_submit,
        "employee_name": employee_name,
        "employee_password": employee_password,
        "employee_submit": employee_submit
    }

def main() -> None:
    """Main function to launch the application."""
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    stacked_widget, widgets = setup_main_window(main_window)
    
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()