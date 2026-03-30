from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import backend

backend.connect_database()

employee_data = None
# Page Constants (for reference)
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
# -------------------------------------------------------------------------------------------------------------
# === Reusable UI Component Functions ===
# -------------------------------------------------------------------------------------------------------------


def create_styled_frame(parent, min_size=None, style=""):
    """Create a styled QFrame with optional minimum size and custom style."""
    frame = QtWidgets.QFrame(parent)
    frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
    frame.setFrameShadow(QtWidgets.QFrame.Raised)
    if min_size:
        frame.setMinimumSize(QtCore.QSize(*min_size))
    frame.setStyleSheet(style)
    return frame


def create_styled_label(
    parent, text, font_size=12, bold=False, style="color: #2c3e50; padding: 10px;"
):
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

    label = create_styled_label(
        frame, label_text, font_size=12, bold=True, style="color: #2c3e50;"
    )
    if min_label_size:
        label.setMinimumSize(QtCore.QSize(*min_label_size))

    line_edit = QtWidgets.QLineEdit(frame)
    line_edit.setFont(FONT_SIZE)
    line_edit.setStyleSheet(
        "background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; padding: 8px;"
    )

    layout.addWidget(label)
    layout.addWidget(line_edit)
    return frame, line_edit


def create_input_field_V(parent, label_text, min_label_size=(120, 0)):
    """Create a horizontal layout with a label and a QLineEdit."""
    frame = create_styled_frame(parent, style="padding: 7px;")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    label = create_styled_label(
        frame, label_text, font_size=12, bold=True, style="color: #2c3e50;"
    )
    if min_label_size:
        label.setMinimumSize(QtCore.QSize(*min_label_size))

    line_edit = QtWidgets.QLineEdit(frame)
    line_edit.setStyleSheet(
        "background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; padding: 8px;"
    )
    line_edit.setFont(FONT_SIZE)

    layout.addWidget(label)
    layout.addWidget(line_edit)
    return frame, line_edit


def show_popup_message(
    parent,
    message: str,
    page: int = None,
    show_cancel: bool = False,
    cancel_page: int = HOME_PAGE,
):
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
            parent.setCurrentIndex(cancel_page)
        dialog.reject()

    button_box.accepted.connect(on_accept)
    button_box.rejected.connect(on_reject)

    dialog.exec_()


def search_result(parent, title, label_text):
    page, main_layout = create_page_with_header(parent, title)
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    content_layout.alignment

    form_frame = create_styled_frame(
        content_frame,
        min_size=(400, 200),
        style="background-color: #ffffff; border-radius: 15px; padding: 10px;",
    )
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(3)
    # Define input fields
    user = create_input_field(form_frame, label_text, min_label_size=(180, 0))
    form_layout.addWidget(user[0])
    user_account_number = user[1]
    user_account_number.setFont(FONT_SIZE)
    submit_button = create_styled_button(form_frame, "Submit", min_size=(100, 50))
    form_layout.addWidget(submit_button)
    content_layout.addWidget(
        form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
    main_layout.addWidget(content_frame)

    return page, (user_account_number, submit_button)


# -------------------------------------------------------------------------------------------------------------
# === Page Creation Functions ==
# -------------------------------------------------------------------------------------------------------------
def create_page_with_header(parent, title_text):
    """Create a page with a styled header and return the page + main layout."""
    page = QtWidgets.QWidget(parent)
    main_layout = QtWidgets.QVBoxLayout(page)
    main_layout.setContentsMargins(20, 20, 20, 20)
    main_layout.setSpacing(20)

    header_frame = create_styled_frame(
        page, style="background-color: #ffffff; border-radius: 10px; padding: 10px;"
    )
    header_layout = QtWidgets.QVBoxLayout(header_frame)
    title_label = create_styled_label(header_frame, title_text, font_size=30)
    header_layout.addWidget(title_label, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

    main_layout.addWidget(header_frame, 0, QtCore.Qt.AlignTop)
    return page, main_layout


def get_employee_name(parent, name_field_text="Enter Employee Name"):
    page, main_layout = create_page_with_header(parent, "Employee Data Update")

    # Content frame
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    # Form frame
    form_frame = create_styled_frame(
        content_frame,
        min_size=(340, 200),
        style="background-color: #ffffff; border-radius: 15px; padding: 10px;",
    )
    form_layout = QtWidgets.QVBoxLayout(form_frame)

    # Form fields
    name_label, name_field = create_input_field(form_frame, name_field_text)
    search_button = create_styled_button(form_frame, "Search", min_size=(100, 30))
    form_layout.addWidget(name_label)
    form_layout.addWidget(search_button)
    content_layout.addWidget(
        form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
    main_layout.addWidget(content_frame)

    def on_search_button_clicked():
        global employee_data
        entered_name = name_field.text().strip()
        print(f"Entered Name: {entered_name}")
        if not entered_name:
            QtWidgets.QMessageBox.warning(
                parent, "Input Error", "Please enter an employee name."
            )
            return

        try:
            employee_check = backend.check_name_in_staff(entered_name)
            print(f"Employee Check: {type(employee_check)},{employee_check}")
            if employee_check:
                cur = backend.cur
                cur.execute("SELECT * FROM staff WHERE name = ?", (entered_name,))
                employee_data = cur.fetchone()
                print(f"Employee Data: {employee_data}")
                parent.setCurrentIndex(UPDATE_EMPLOYEE_PAGE2)

            # if employee_data:
            # QtWidgets.QMessageBox.information(parent, "Employee Found",
            #                                   f"Employee data:\nID: {fetch[0]}\nName: {fetch[1]}\nDept: {fetch[2]}\nRole: {fetch[3]}")

            else:
                QtWidgets.QMessageBox.information(
                    parent, "Not Found", "Employee not found."
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                parent, "Error", f"An error occurred: {str(e)}"
            )

    search_button.clicked.connect(on_search_button_clicked)

    return page

    # backend.check_name_in_staff()


def create_login_page(
    parent,
    title,
    name_field_text="Name :",
    password_field_text="Password :",
    submit_text="Submit",
):
    """Create a login page with a title, name and password fields, and a submit button."""
    page, main_layout = create_page_with_header(parent, title)

    # Content frame
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    # Form frame
    form_frame = create_styled_frame(
        content_frame,
        min_size=(340, 200),
        style="background-color: #ffffff; border-radius: 15px; padding: 10px;",
    )
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

    content_layout.addWidget(
        form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
    main_layout.addWidget(content_frame)

    return page, name_edit, password_edit, submit_button


def on_login_button_clicked(parent, name_field, password_field):
    name = name_field.text().strip()
    password = password_field.text().strip()

    if not name or not password:
        show_popup_message(parent, "Please enter your name and password.", HOME_PAGE)
    else:
        try:
            # Ideally, here you'd call a backend authentication check
            success = backend.check_admin(name, password)
            if success:
                QtWidgets.QMessageBox.information(
                    parent, "Login Successful", f"Welcome, {name}!"
                )
            else:
                QtWidgets.QMessageBox.warning(
                    parent, "Login Failed", "Incorrect name or password."
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                parent, "Error", f"An error occurred during login: {str(e)}"
            )


def create_home_page(parent, on_admin_clicked, on_employee_clicked, on_exit_clicked):
    """Create the home page with Admin, Employee, and Exit buttons."""
    page, main_layout = create_page_with_header(parent, "Admin Menu")

    # Button frame
    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    button_layout = QtWidgets.QVBoxLayout(button_frame)

    # Button container
    button_container = create_styled_frame(
        button_frame,
        min_size=(300, 0),
        style="background-color: #ffffff; border-radius: 15px; padding: 20px;",
    )
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

    button_layout.addWidget(
        button_container, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
    main_layout.addWidget(button_frame)

    # Connect button signals
    admin_button.clicked.connect(on_admin_clicked)
    employee_button.clicked.connect(on_employee_clicked)
    exit_button.clicked.connect(on_exit_clicked)

    return page


def create_admin_menu_page(parent):
    page, main_layout = create_page_with_header(parent, "Admin Menu")

    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    button_layout = QtWidgets.QVBoxLayout(button_frame)

    button_container = create_styled_frame(
        button_frame,
        min_size=(300, 0),
        style="background-color: #ffffff; border-radius: 15px; padding: 20px;",
    )
    button_container_layout = QtWidgets.QVBoxLayout(button_container)
    button_container_layout.setSpacing(15)

    # Define button labels
    button_labels = [
        "Add Employee",
        "Update Employee",
        "Employee List",
        "Total Money",
        "Back",
    ]
    buttons = []

    for label in button_labels:
        btn = create_styled_button(button_container, label)
        button_container_layout.addWidget(btn)
        buttons.append(btn)

    button_layout.addWidget(
        button_container, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
    main_layout.addWidget(button_frame)

    return page, *buttons  # Unpack as add_button, update_employee, etc.


def create_add_employee_page(
    parent, title, submit_text="Submit", update_btn: bool = False
):
    page, main_layout = create_page_with_header(parent, title)

    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    form_frame = create_styled_frame(
        content_frame,
        min_size=(340, 200),
        style="background-color: #ffffff; border-radius: 15px; padding: 10px;",
    )
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(10)

    # Define input fields
    fields = ["Name :", "Password :", "Salary :", "Position :"]
    name_edit = None
    password_edit = None
    salary_edit = None
    position_edit = None
    edits = []

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
        edits.append(field_edit)
    # Submit button
    button_frame = create_styled_frame(form_frame, style="padding: 7px;")
    button_layout = QtWidgets.QVBoxLayout(button_frame)
    if update_btn:
        update_button = create_styled_button(button_frame, "Update", min_size=(100, 50))
        button_layout.addWidget(update_button, 0, QtCore.Qt.AlignHCenter)
    else:
        submit_button = create_styled_button(
            button_frame, submit_text, min_size=(100, 50)
        )
        button_layout.addWidget(submit_button, 0, QtCore.Qt.AlignHCenter)

    form_layout.addWidget(button_frame)
    content_layout.addWidget(
        form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
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
    main_layout.addWidget(back_btn, 0, alignment=QtCore.Qt.AlignLeft)
    if update_btn:
        return page, name_edit, password_edit, salary_edit, position_edit, update_button
    else:
        return (
            page,
            name_edit,
            password_edit,
            salary_edit,
            position_edit,
            submit_button,
        )  # Unpack as name_edit, password_edit, etc.


def show_employee_list_page(parent, title):
    page, main_layout = create_page_with_header(parent, title)

    content_frame = create_styled_frame(
        page, style="background-color: #f9f9f9; border-radius: 10px; padding: 15px;"
    )
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    # Table frame
    table_frame = create_styled_frame(
        content_frame,
        style="background-color: #ffffff;  border-radius: 8px; padding: 10px;",
    )
    table_layout = QtWidgets.QVBoxLayout(table_frame)
    table_layout.setSpacing(0)

    # Header row
    header_frame = create_styled_frame(
        table_frame,
        style="background-color: #f5f5f5; ; border-radius: 8px 8px 0 0; padding: 10px;",
    )
    header_layout = QtWidgets.QHBoxLayout(header_frame)
    header_layout.setContentsMargins(10, 5, 10, 5)
    headers = ["Name", "Position", "Salary"]
    for i, header in enumerate(headers):
        header_label = QtWidgets.QLabel(header, header_frame)
        header_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #333333; padding: 0px; margin: 0px;"
        )
        if i == 2:  # Right-align salary header
            header_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        else:
            header_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        header_layout.addWidget(
            header_label, 1 if i < 2 else 0
        )  # Stretch name and position, not salary
    table_layout.addWidget(header_frame)

    # Employee rows
    employees = backend.show_employees_for_update()
    for row, employee in enumerate(employees):
        row_frame = create_styled_frame(
            table_frame,
            style=f"background-color: {'#fafafa' if row % 2 else '#ffffff'}; padding: 8px;",
        )
        row_layout = QtWidgets.QHBoxLayout(row_frame)
        row_layout.setContentsMargins(10, 5, 10, 5)

        # Name
        name_label = QtWidgets.QLabel(employee[0], row_frame)
        name_label.setStyleSheet(
            "font-size: 14px; color: #333333; padding: 0px; margin: 0px;"
        )
        name_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        row_layout.addWidget(name_label, 1)

        # Position
        position_label = QtWidgets.QLabel(employee[3], row_frame)
        position_label.setStyleSheet(
            "font-size: 14px; color: #333333; padding: 0px; margin: 0px;"
        )
        position_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        row_layout.addWidget(position_label, 1)

        # Salary (formatted as currency)
        salary_label = QtWidgets.QLabel(f"${float(employee[2]):,.2f}", row_frame)
        salary_label.setStyleSheet(
            "font-size: 14px; color: #333333; padding: 0px; margin: 0px;"
        )
        salary_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        row_layout.addWidget(salary_label, 0)

        table_layout.addWidget(row_frame)

    # Add stretch to prevent rows from expanding vertically
    table_layout.addStretch()

    # Back button
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


def show_total_money(parent, title):
    page, main_layout = create_page_with_header(parent, title)

    content_frame = create_styled_frame(
        page, style="background-color: #f9f9f9; border-radius: 10px; padding: 15px;"
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    content_layout.setProperty("spacing", 10)
    all = backend.all_money()

    # Total money label
    total_money_label = QtWidgets.QLabel(f"Total Money: ${all}", content_frame)
    total_money_label.setStyleSheet(
        "font-size: 24px; font-weight: bold; color: #333333;"
    )
    content_layout.addWidget(total_money_label, alignment=QtCore.Qt.AlignCenter)
    # Back button
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


# -----------employees menu pages-----------
def create_employee_menu_page(parent, title):
    page, main_layout = create_page_with_header(parent, title)

    button_frame = create_styled_frame(page)
    button_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    button_layout = QtWidgets.QVBoxLayout(button_frame)

    button_container = create_styled_frame(
        button_frame,
        min_size=(300, 0),
        style="background-color: #ffffff; border-radius: 15px; padding: 20px;",
    )
    button_container_layout = QtWidgets.QVBoxLayout(button_container)
    button_container_layout.setSpacing(15)

    # Define button labels
    button_labels = [
        "Create Account ",
        "Show Details",
        "Add Balance",
        "Withdraw Money",
        "Chack Balanace",
        "Update Account",
        "list of all Members",
        "Delete Account",
        "Back",
    ]
    buttons = []

    for label in button_labels:
        btn: QtWidgets.QPushButton = create_styled_button(button_container, label)
        button_container_layout.addWidget(btn)
        buttons.append(btn)

    button_layout.addWidget(
        button_container, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
    main_layout.addWidget(button_frame)

    return page, *buttons  # Unpack as add_button, update_employee, etc.


def create_account_page(parent, title, update_btn=False):
    page, main_layout = create_page_with_header(parent, title)

    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    form_frame = create_styled_frame(
        content_frame,
        min_size=(400, 200),
        style="background-color: #ffffff; border-radius: 15px; padding: 10px;",
    )
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(3)

    # Define input fields
    fields = ["Name :", "Age :", "Address", "Balance :", "Mobile number :"]
    edits = []

    for i, field in enumerate(fields):
        field_frame, field_edit = create_input_field(
            form_frame, field, min_label_size=(160, 0)
        )
        form_layout.addWidget(field_frame)
        field_edit.setFont(QtGui.QFont("Arial", 12))
        if i == 0:
            name_edit = field_edit
        elif i == 1:
            Age_edit = field_edit
        elif i == 2:
            Address_edit = field_edit
        elif i == 3:
            Balance_edit = field_edit
        elif i == 4:
            Mobile_number_edit = field_edit
        edits.append(field_edit)
    # Dropdown for account type
    account_type_label = QtWidgets.QLabel("Account Type :", form_frame)
    account_type_label.setStyleSheet(
        "font-size: 14px; font-weight: bold; color: #333333;"
    )
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
        QComboBox::down-arrow {
            width: 12px;
            height: 12px;
        }
        QComboBox QAbstractItemView {
            border: 1px solid #ccc;
            background-color: white;
            selection-background-color: #0078d4;
            selection-color: white;
        }
    """)
    form_layout.addWidget(account_type_dropdown)

    # Submit button
    button_frame = create_styled_frame(form_frame, style="padding: 7px;")
    button_layout = QtWidgets.QVBoxLayout(button_frame)

    if update_btn:
        submit_button = create_styled_button(button_frame, "Update", min_size=(100, 50))
    else:
        submit_button = create_styled_button(button_frame, "Submit", min_size=(100, 50))
    button_layout.addWidget(submit_button, 0, QtCore.Qt.AlignHCenter)

    form_layout.addWidget(button_frame)
    content_layout.addWidget(
        form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
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
    main_layout.addWidget(back_btn, 0, alignment=QtCore.Qt.AlignLeft)

    return page, (
        name_edit,
        Age_edit,
        Address_edit,
        Balance_edit,
        Mobile_number_edit,
        account_type_dropdown,
        submit_button,
    )


def create_show_details_page1(parent, title):
    page, main_layout = create_page_with_header(parent, title)
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    form_frame = create_styled_frame(
        content_frame,
        min_size=(400, 200),
        style="background-color: #ffffff; border-radius: 15px; padding: 10px;",
    )
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(3)
    # Define input fields
    bannk_user = create_input_field(
        form_frame, "Enter Bank account Number :", min_label_size=(180, 0)
    )
    form_layout.addWidget(bannk_user[0])
    user_account_number = bannk_user[1]
    submit_button = create_styled_button(form_frame, "Submit", min_size=(100, 50))
    form_layout.addWidget(submit_button)
    content_layout.addWidget(
        form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
    main_layout.addWidget(content_frame)

    return page, (user_account_number, submit_button)


def create_show_details_page2(parent, title):
    page, main_layout = create_page_with_header(parent, title)
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)

    form_frame = create_styled_frame(
        content_frame,
        min_size=(400, 200),
        style="background-color: #ffffff; border-radius: 15px; padding: 10px;",
    )
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    form_layout.setSpacing(3)

    # Define input fields

    labeles = [
        "Account No: ",
        "Name: ",
        "Age:",
        "Address: ",
        "Balance: ",
        "Mobile Number: ",
        "Account Type: ",
    ]
    for i in range(len(labeles)):
        label_frame, input_field = create_input_field(
            form_frame, labeles[i], min_label_size=(180, 30)
        )
        form_layout.addWidget(label_frame)
        input_field.setReadOnly(True)
        input_field.setFont(QtGui.QFont("Arial", 12))
        if i == 0:
            account_no_field = input_field
        elif i == 1:
            name_field = input_field
        elif i == 2:
            age_field = input_field
        elif i == 3:
            address_field = input_field
        elif i == 4:
            balance_field = input_field
        elif i == 5:
            mobile_number_field = input_field
        elif i == 6:
            account_type_field = input_field

    exite_btn = create_styled_button(form_frame, "Exit", min_size=(100, 50))
    exite_btn.setStyleSheet("""
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
    exite_btn.clicked.connect(lambda: parent.setCurrentIndex(EMPLOYEE_MENU_PAGE))
    content_layout.addWidget(
        form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
    main_layout.addWidget(content_frame)
    main_layout.addWidget(exite_btn)

    return page, (
        account_no_field,
        name_field,
        age_field,
        address_field,
        balance_field,
        mobile_number_field,
        account_type_field,
        exite_btn,
    )


def update_user(parent, title, input_fields_label, input_fielf: bool = True):
    page, main_layout = create_page_with_header(parent, title)
    content_frame = create_styled_frame(page)
    content_frame.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
    )
    content_layout = QtWidgets.QVBoxLayout(content_frame)
    content_layout.alignment

    form_frame = create_styled_frame(
        content_frame,
        min_size=(400, 200),
        style="background-color: #ffffff; border-radius: 15px; padding: 10px;",
    )
    form_layout = QtWidgets.QVBoxLayout(form_frame)
    form_layout.setSpacing(3)
    # Define input fields
    user = create_input_field(form_frame, "User Name: ", min_label_size=(180, 0))
    user_balance = create_input_field(form_frame, "Balance: ", min_label_size=(180, 0))

    # Add input fields to the form layout
    form_layout.addWidget(user[0])
    form_layout.addWidget(user_balance[0])
    if input_fielf:
        user_update_balance = create_input_field_V(
            form_frame, input_fields_label, min_label_size=(180, 0)
        )
        form_layout.addWidget(user_update_balance[0])

    # Store the input fields in variables
    user_account_name = user[1]
    user_account_name.setReadOnly(True)
    user_account_name.setStyleSheet(
        "background-color: #8a8a8a; border: 1px solid #ccc; border-radius: 4px; padding: 8px;"
    )
    user_balance_field = user_balance[1]
    user_balance_field.setReadOnly(True)
    user_balance_field.setStyleSheet(
        "background-color: #8a8a8a; border: 1px solid #ccc; border-radius: 4px; padding: 8px;"
    )
    if input_fielf:
        user_update_balance_field = user_update_balance[1]
        user_update_balance_field.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; padding: 8px;"
        )

    # Set the font size for the input fields
    user_account_name.setFont(FONT_SIZE)
    user_balance_field.setFont(FONT_SIZE)
    if input_fielf:
        user_update_balance_field.setFont(FONT_SIZE)

    # Add a submit button
    submit_button = create_styled_button(form_frame, "Submit", min_size=(100, 50))
    form_layout.addWidget(submit_button)
    content_layout.addWidget(
        form_frame, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
    )
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
    backend
    if input_fielf:
        return page, (
            user_account_name,
            user_balance_field,
            user_update_balance_field,
            submit_button,
        )
    else:
        return page, (user_account_name, user_balance_field, submit_button)


# -------------------------------------------------------------------------------------------------------------
# === Main Window Setup ===
# -------------------------------------------------------------------------------------------------------------


def setup_main_window(main_window: QtWidgets.QMainWindow):
    """Set up the main window with a stacked widget containing home, admin, and employee pages."""
    main_window.setObjectName("MainWindow")
    main_window.resize(800, 600)
    main_window.setStyleSheet("background-color: #f0f2f5;")

    central_widget = QtWidgets.QWidget(main_window)
    main_layout = QtWidgets.QHBoxLayout(central_widget)

    stacked_widget = QtWidgets.QStackedWidget(central_widget)

    # Create pages
    def switch_to_admin():
        stacked_widget.setCurrentIndex(ADMIN_PAGE)

    def switch_to_employee():
        stacked_widget.setCurrentIndex(EMPLOYEE_PAGE)

    def exit_app():
        QtWidgets.QApplication.quit()

    def admin_login_menu_page(name, password):
        try:
            # Ideally, here you'd call a backend authentication check
            success = backend.check_admin(name, password)
            if success:
                QtWidgets.QMessageBox.information(
                    stacked_widget, "Login Successful", f"Welcome, {name}!"
                )
                stacked_widget.setCurrentIndex(ADMIN_MENU_PAGE)
            else:
                QtWidgets.QMessageBox.warning(
                    stacked_widget, "Login Failed", "Incorrect name or password."
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                stacked_widget, "Error", f"An error occurred during login: {str(e)}"
            )
            # show_popup_message(stacked_widget,"Invalid admin credentials",0)

    def add_employee_form_submit(name, password, salary, position):
        if (
            len(name) != 0
            and len(password) != 0
            and len(salary) != 0
            and len(position) != 0
        ):
            backend.create_employee(name, password, salary, position)
            show_popup_message(
                stacked_widget, "Employee added successfully", ADMIN_MENU_PAGE
            )

        else:
            print("Please fill in all fields")
            show_popup_message(
                stacked_widget, "Please fill in all fields", ADD_EMPLOYEE_PAGE
            )

    def update_employee_data(name, password, salary, position, name_to_update):
        try:
            cur = backend.cur
            if name_to_update:
                cur.execute(
                    "UPDATE staff SET Name = ? WHERE name = ?", (name, name_to_update)
                )

            cur.execute("UPDATE staff SET Name = ? WHERE name = ?", (password, name))
            cur.execute(
                "UPDATE staff SET password = ? WHERE name = ?", (password, name)
            )
            cur.execute("UPDATE staff SET salary = ? WHERE name = ?", (salary, name))
            cur.execute(
                "UPDATE staff SET position = ? WHERE name = ?", (position, name)
            )
            backend.conn.commit()
            show_popup_message(
                stacked_widget, "Employee Update successfully", UPDATE_EMPLOYEE_PAGE2
            )

        except:
            show_popup_message(
                stacked_widget, "Please fill in all fields", UPDATE_EMPLOYEE_PAGE2
            )

    # Create Home Page
    home_page = create_home_page(
        stacked_widget, switch_to_admin, switch_to_employee, exit_app
    )
    # ------------------------------------------------------------------------------------------------
    # -------------------------------------Admin panel page ---------------------------------------
    # ------------------------------------------------------------------------------------------------
    # Create Admin Login Page
    admin_page, admin_name, admin_password, admin_submit = create_login_page(
        stacked_widget, title="Admin Login"
    )
    admin_password.setEchoMode(QtWidgets.QLineEdit.Password)
    admin_name.setFont(QtGui.QFont("Arial", 10))
    admin_password.setFont(QtGui.QFont("Arial", 10))
    admin_name.setPlaceholderText("Enter your name")
    admin_password.setPlaceholderText("Enter your password")

    admin_submit.clicked.connect(
        lambda: admin_login_menu_page(admin_name.text(), admin_password.text())
    )

    # Create Admin Menu Page
    (
        admin_menu_page,
        add_button,
        update_button,
        list_button,
        money_button,
        back_button,
    ) = create_admin_menu_page(stacked_widget)

    add_button.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(ADD_EMPLOYEE_PAGE)
    )
    update_button.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(UPDATE_EMPLOYEE_PAGE1)
    )
    list_button.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(EMPLOYEE_LIST_PAGE)
    )
    back_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(HOME_PAGE))
    money_button.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(ADMIN_TOTAL_MONEY)
    )
    # Create Add Employee Page
    add_employee_page, emp_name, emp_password, emp_salary, emp_position, emp_submit = (
        create_add_employee_page(stacked_widget, title="Add Employee")
    )

    # Update Employee Page
    u_employee_page1 = get_employee_name(stacked_widget)
    # apply the update_employee_data function to the submit button

    (
        u_employee_page2,
        u_employee_name,
        u_employee_password,
        u_employee_salary,
        u_employee_position,
        u_employee_update,
    ) = create_add_employee_page(
        stacked_widget, "Update Employee Details", update_btn=True
    )

    def populate_employee_data():
        global employee_data
        if employee_data:
            print("employee_data is not None")
            u_employee_name.setText(str(employee_data[0]))  # Name
            u_employee_password.setText(str(employee_data[1]))  # Password
            u_employee_salary.setText(str(employee_data[2]))  # Salary
            u_employee_position.setText(str(employee_data[3]))  # Position
        else:
            # Clear fields if no employee data is available
            print("employee_data is None")
            u_employee_name.clear()
            u_employee_password.clear()
            u_employee_salary.clear()
            u_employee_position.clear()
            QtWidgets.QMessageBox.warning(
                stacked_widget, "No Data", "No employee data available to display."
            )

    def on_page_changed(index):
        if index == 6:  # update_employee_page2 is at index 6
            populate_employee_data()

    # Connect the currentChanged signal to the on_page_changed function
    stacked_widget.currentChanged.connect(on_page_changed)

    def update_employee_data(name, password, salary, position, name_to_update):
        try:
            if not name_to_update:
                show_popup_message(
                    stacked_widget,
                    "Original employee name is missing.",
                    UPDATE_EMPLOYEE_PAGE2,
                )
                return
            if not (name or password or salary or position):
                show_popup_message(
                    stacked_widget,
                    "Please fill at least one field to update.",
                    UPDATE_EMPLOYEE_PAGE2,
                )
                return
            if name:
                backend.update_employee_name(name, name_to_update)
            if password:
                backend.update_employee_password(password, name_to_update)
            if salary:
                try:
                    salary = int(salary)
                    backend.update_employee_salary(salary, name_to_update)
                except ValueError:
                    show_popup_message(
                        stacked_widget, "Salary must be a valid number.", 5
                    )
                    return
            if position:
                backend.update_employee_position(position, name_to_update)
            show_popup_message(
                stacked_widget, "Employee updated successfully.", ADMIN_MENU_PAGE
            )
        except Exception as e:
            show_popup_message(
                stacked_widget,
                f"Error updating employee: {str(e)}",
                UPDATE_EMPLOYEE_PAGE2,
                show_cancel=True,
                cancel_page=ADMIN_MENU_PAGE,
            )

    u_employee_update.clicked.connect(
        lambda: update_employee_data(
            u_employee_name.text().strip(),
            u_employee_password.text().strip(),
            u_employee_salary.text().strip(),
            u_employee_position.text().strip(),
            employee_data[0] if employee_data else "",
        )
    )

    emp_submit.clicked.connect(
        lambda: add_employee_form_submit(
            emp_name.text(), emp_password.text(), emp_salary.text(), emp_position.text()
        )
    )
    # show employee list page
    employee_list_page = show_employee_list_page(stacked_widget, "Employee List")
    admin_total_money = show_total_money(stacked_widget, "Total Money")
    # ------------------------------------------------------------------------------------------------
    # -------------------------------------Employee panel page ---------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Create Employee Login Page
    employee_page, employee_name, employee_password, employee_submit = (
        create_login_page(stacked_widget, title="Employee Login")
    )
    employee_submit.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(EMPLOYEE_MENU_PAGE)
    )
    (
        employee_menu_page,
        E_Create_Account,
        E_Show_Details,
        E_add_Balance,
        E_Withdraw_Money,
        E_Chack_Balanace,
        E_Update_Account,
        E_list_of_all_Members,
        E_Delete_Account,
        E_Back,
    ) = create_employee_menu_page(stacked_widget, "Employee Menu")
    # List of all  page
    E_Create_Account.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(EMPLOYEE_CREATE_ACCOUNT_PAGE)
    )
    E_Show_Details.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(EMPLOYEE_SHOW_DETAILS_PAGE1)
    )
    E_add_Balance.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(EMPLOYEE_ADD_BALANCE_SEARCH)
    )
    E_Withdraw_Money.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(EMPLOYEE_WITHDRAW_MONEY_SEARCH)
    )
    E_Chack_Balanace.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(EMPLOYEE_CHECK_BALANCE_SEARCH)
    )
    E_Update_Account.clicked.connect(
        lambda: stacked_widget.setCurrentIndex(EMPLOYEE_UPDATE_ACCOUNT_SEARCH)
    )
    # E_list_of_all_Members.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_LIST_OF_ALL_MEMBERS_PAGE))
    # E_Delete_Account.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_DELETE_ACCOUNT_PAGE))
    # E_Back.clicked.connect(lambda: stacked_widget.setCurrentIndex(EMPLOYEE_MENU_PAGE))

    employee_create_account_page, all_employee_menu_btn = create_account_page(
        stacked_widget, "Create Account"
    )
    all_employee_menu_btn[6].clicked.connect(
        lambda: add_account_form_submit(
            all_employee_menu_btn[0].text().strip(),
            all_employee_menu_btn[1].text().strip(),
            all_employee_menu_btn[2].text().strip(),
            all_employee_menu_btn[3].text().strip(),
            all_employee_menu_btn[5].currentText(),
            all_employee_menu_btn[4].text().strip(),
        )
    )

    def add_account_form_submit(name, age, address, balance, account_type, mobile):
        if (
            len(name) != 0
            and len(age) != 0
            and len(address) != 0
            and len(balance) != 0
            and len(account_type) != 0
            and len(mobile) != 0
        ):
            try:
                balance = int(balance)
            except ValueError:
                show_popup_message(
                    stacked_widget,
                    "Balance must be a valid number",
                    EMPLOYEE_CREATE_ACCOUNT_PAGE,
                )
                return
            if balance < 0:
                show_popup_message(
                    stacked_widget,
                    "Balance cannot be negative",
                    EMPLOYEE_CREATE_ACCOUNT_PAGE,
                )
                return
            if account_type not in ["Savings", "Current", "Fixed Deposit"]:
                show_popup_message(
                    stacked_widget, "Invalid account type", EMPLOYEE_CREATE_ACCOUNT_PAGE
                )
                return
            if len(mobile) != 10:
                show_popup_message(
                    stacked_widget,
                    "Mobile number must be 10 digits",
                    EMPLOYEE_CREATE_ACCOUNT_PAGE,
                )
                return
            if not mobile.isdigit():
                show_popup_message(
                    stacked_widget,
                    "Mobile number must contain only digits",
                    EMPLOYEE_CREATE_ACCOUNT_PAGE,
                )
                return
            if not name.isalpha():
                show_popup_message(
                    stacked_widget,
                    "Name must contain only alphabets",
                    EMPLOYEE_CREATE_ACCOUNT_PAGE,
                )
                return
            if not age.isdigit():
                show_popup_message(
                    stacked_widget,
                    "Age must contain only digits",
                    EMPLOYEE_CREATE_ACCOUNT_PAGE,
                )
                return
            if int(age) < 18:
                show_popup_message(
                    stacked_widget,
                    "Age must be greater than 18",
                    EMPLOYEE_CREATE_ACCOUNT_PAGE,
                )
                return
            if len(address) < 10:
                show_popup_message(
                    stacked_widget,
                    "Address must be at least 10 characters long",
                    EMPLOYEE_CREATE_ACCOUNT_PAGE,
                )
                return
            backend.create_customer(name, age, address, balance, account_type, mobile)
            all_employee_menu_btn[0].setText("")
            all_employee_menu_btn[1].setText("")
            all_employee_menu_btn[2].setText("")
            all_employee_menu_btn[3].setText("")
            all_employee_menu_btn[4].setText("")
            (all_employee_menu_btn[5].currentText(),)
            show_popup_message(
                stacked_widget,
                "Account created successfully",
                EMPLOYEE_MENU_PAGE,
                False,
            )
        else:
            show_popup_message(
                stacked_widget,
                "Please fill in all fields",
                EMPLOYEE_CREATE_ACCOUNT_PAGE,
            )
            # Add pages to stacked widget

    show_bank_user_data_page1, show_bank_user_other1 = create_show_details_page1(
        stacked_widget, "Show Details"
    )
    show_bank_user_data_page2, show_bank_user_other2 = create_show_details_page2(
        stacked_widget, "Show Details"
    )

    show_bank_user_other1[1].clicked.connect(
        lambda: show_bank_user_data_page1_submit_btn(
            int(show_bank_user_other1[0].text().strip())
        )
    )

    def show_bank_user_data_page1_submit_btn(name: int):
        account_data = backend.get_details(name)
        if account_data:
            show_bank_user_other1[0].setText("")
            show_bank_user_other2[0].setText(str(account_data[0]))
            show_bank_user_other2[1].setText(str(account_data[1]))
            show_bank_user_other2[2].setText(str(account_data[2]))
            show_bank_user_other2[3].setText(str(account_data[3]))
            show_bank_user_other2[4].setText(str(account_data[4]))
            show_bank_user_other2[5].setText(str(account_data[5]))
            show_bank_user_other2[6].setText(str(account_data[6]))
            stacked_widget.setCurrentIndex(EMPLOYEE_SHOW_DETAILS_PAGE2)
        else:
            show_popup_message(
                stacked_widget, "Account not found", EMPLOYEE_SHOW_DETAILS_PAGE1
            )

    def setup_balance_operation_flow(
        stacked_widget,
        title_search,
        placeholder,
        title_form,
        action_button_text,
        success_message,
        backend_action_fn,
        stacked_page_index,
        search_index,
        page_index,
        need_input=True,
    ):
        # Create search UI
        search_page, search_widgets = search_result(
            stacked_widget, title_search, placeholder
        )
        search_input = search_widgets[0]
        search_button = search_widgets[1]

        # Create update UI
        form_page, form_widgets = update_user(
            stacked_widget, title_form, action_button_text, need_input
        )
        if need_input:
            name_field, balance_field, amount_field, action_button = form_widgets
        else:
            name_field, balance_field, action_button = form_widgets

        def on_search_submit():
            try:
                account_number = int(search_input.text().strip())
            except ValueError:
                show_popup_message(
                    stacked_widget, "Please enter a valid account number.", search_index
                )
                return

            if backend.check_acc_no(account_number):
                account_data = backend.get_details(account_number)
                name_field.setText(str(account_data[1]))
                balance_field.setText(str(account_data[4]))
                stacked_widget.setCurrentIndex(page_index)
            else:
                show_popup_message(
                    stacked_widget,
                    "Account not found",
                    search_index,
                    show_cancel=True,
                    cancel_page=EMPLOYEE_MENU_PAGE,
                )

        def on_action_submit():
            try:
                account_number = int(search_input.text().strip())
                amount = int(amount_field.text().strip())
                backend_action_fn(amount, account_number)
                name_field.setText("")
                balance_field.setText("")
                search_input.setText("")
                show_popup_message(stacked_widget, success_message, EMPLOYEE_MENU_PAGE)
            except ValueError:
                show_popup_message(
                    stacked_widget, "Enter valid numeric amount.", page_index
                )

        search_button.clicked.connect(on_search_submit)
        action_button.clicked.connect(on_action_submit)

        return search_page, form_page

    # Add Balance Flow
    add_balance_search_page, add_balance_page = setup_balance_operation_flow(
        stacked_widget=stacked_widget,
        title_search="Add Balance",
        placeholder="Enter Account Number: ",
        title_form="Add Balance User Account",
        action_button_text="Enter Amount: ",
        success_message="Balance updated successfully",
        backend_action_fn=backend.update_balance,
        stacked_page_index=EMPLOYEE_ADD_BALANCE_SEARCH,
        search_index=EMPLOYEE_ADD_BALANCE_SEARCH,
        page_index=EMPLOYEE_ADD_BALANCE_PAGE,
    )

    # Withdraw Money Flow
    withdraw_money_search_page, withdraw_money_page = setup_balance_operation_flow(
        stacked_widget=stacked_widget,
        title_search="Withdraw Money",
        placeholder="Enter Account Number: ",
        title_form="Withdraw Money From User Account",
        action_button_text="Withdraw Amount: ",
        success_message="Amount withdrawn successfully",
        backend_action_fn=backend.deduct_balance,
        stacked_page_index=EMPLOYEE_WITHDRAW_MONEY_SEARCH,
        search_index=EMPLOYEE_WITHDRAW_MONEY_SEARCH,
        page_index=EMPLOYEE_WITHDRAW_MONEY_PAGE,
    )

    check_balance_search_page, check_balance_page = setup_balance_operation_flow(
        stacked_widget=stacked_widget,
        title_search="Check Balance",
        placeholder="Enter Account Number: ",
        title_form="Check Balance",
        action_button_text="Check Balance: ",
        success_message="Balance checked successfully",
        backend_action_fn=backend.check_balance,
        stacked_page_index=EMPLOYEE_CHECK_BALANCE_SEARCH,
        search_index=EMPLOYEE_CHECK_BALANCE_SEARCH,
        page_index=EMPLOYEE_CHECK_BALANCE_PAGE,
        need_input=False,
    )

    def find_and_hide_submit_button(page):
        # Find all QPushButton widgets in the page
        buttons = page.findChildren(QtWidgets.QPushButton)
        for button in buttons:
            if button.text() == "Submit":
                button.hide()
                break

    find_and_hide_submit_button(check_balance_page)

    # Update Employee details
    update_empolyee_search_page, update_empolyee_search_other = search_result(
        stacked_widget, "Update Employee Details", "Enter Employee ID: "
    )
    update_employee_page, update_employee_other = create_account_page(
        stacked_widget, "Update Employee", True
    )
    name_edit = update_employee_other[0]
    Age_edit = update_employee_other[1]
    Address_edit = update_employee_other[2]
    Balance_edit = update_employee_other[3]
    Mobile_number_edit = update_employee_other[4]
    account_type_dropdown = update_employee_other[5]
    # name_edit, Age_edit,Address_edit,Balance_edit,Mobile_number_edit, account_type_dropdown ,submit_button

    update_empolyee_search_other[1].clicked.connect(
        lambda: update_employee_search_submit()
    )
    update_employee_other[6].clicked.connect(lambda: update_employee_submit())

    def update_employee_search_submit():
        try:
            user_data = backend.get_details(
                int(update_empolyee_search_other[0].text().strip())
            )
            print("Featch data: ", user_data)
            name_edit.setText(str(user_data[1]))
            Age_edit.setText(str(user_data[2]))
            Address_edit.setText(str(user_data[3]))
            Balance_edit.setText(str(user_data[4]))
            Mobile_number_edit.setText(str(user_data[6]))
            Balance_edit.setDisabled(True)
            account_type_dropdown.setCurrentText(str(user_data[5]))
            stacked_widget.setCurrentIndex(EMPLOYEE_UPDATE_ACCOUNT_PAGE)
        except ValueError:
            show_popup_message(
                stacked_widget, "Enter valid numeric employee ID.", EMPLOYEE_MENU_PAGE
            )

    def update_employee_submit():
        try:
            user_data = backend.get_details(
                int(update_empolyee_search_other[0].text().strip())
            )
            name = name_edit.text().strip()
            age = int(Age_edit.text().strip())
            address = Address_edit.text().strip()
            mobile_number = int(Mobile_number_edit.text().strip())
            account_type = account_type_dropdown.currentText()
            print(name, age, address, mobile_number, account_type)
            backend.update_name_in_bank_table(name, user_data[0])
            backend.update_age_in_bank_table(age, user_data[0])
            backend.update_address_in_bank_table(address, user_data[0])
            backend.update_address_in_bank_table(address, user_data[0])
            backend.update_mobile_number_in_bank_table(mobile_number, user_data[0])
            backend.update_acc_type_in_bank_table(account_type, user_data[0])

            show_popup_message(
                stacked_widget,
                "Employee details updated successfully",
                EMPLOYEE_MENU_PAGE,
            )
            stacked_widget.setCurrentIndex(EMPLOYEE_MENU_PAGE)
        except ValueError as e:
            print(e)
            show_popup_message(
                stacked_widget, "Enter valid numeric employee ID.", EMPLOYEE_MENU_PAGE
            )

    stacked_widget.addWidget(home_page)  # 0
    stacked_widget.addWidget(admin_page)  # 1
    stacked_widget.addWidget(employee_page)  # 2
    stacked_widget.addWidget(admin_menu_page)  # 3
    stacked_widget.addWidget(add_employee_page)  # 4
    stacked_widget.addWidget(u_employee_page1)  # 5
    stacked_widget.addWidget(u_employee_page2)  # 6
    stacked_widget.addWidget(employee_list_page)  # 7
    stacked_widget.addWidget(admin_total_money)  # 8
    stacked_widget.addWidget(employee_menu_page)  # 9
    stacked_widget.addWidget(employee_create_account_page)  # 10
    stacked_widget.addWidget(show_bank_user_data_page1)  # 11
    stacked_widget.addWidget(show_bank_user_data_page2)  # 12
    stacked_widget.addWidget(add_balance_search_page)  # 13
    stacked_widget.addWidget(add_balance_page)  # 14
    stacked_widget.addWidget(withdraw_money_search_page)  # 15
    stacked_widget.addWidget(withdraw_money_page)  # 16
    stacked_widget.addWidget(check_balance_search_page)  # 17
    stacked_widget.addWidget(check_balance_page)  # 18
    stacked_widget.addWidget(update_empolyee_search_page)  # 19
    stacked_widget.addWidget(update_employee_page)  # 20

    main_layout.addWidget(stacked_widget)
    main_window.setCentralWidget(central_widget)

    # Set initial page
    stacked_widget.setCurrentIndex(9)

    return stacked_widget, {
        "admin_name": admin_name,
        "admin_password": admin_password,
        "admin_submit": admin_submit,
        "employee_name": employee_name,
        "employee_password": employee_password,
        "employee_submit": employee_submit,
    }


def main():
    """Main function to launch the application."""
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    stacked_widget, widgets = setup_main_window(main_window)

    # Example: Connect submit buttons to print input values

    main_window.show()
    sys.exit(app.exec_())


# -------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# TO-DO:
# 1.refese the employee list page after add or delete or update employee
