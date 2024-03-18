# RobofestWarehouseRobot

[comment]: # (<img src="/imgs/turbotoster_team.svg" alt="team image" width="50%"/>)

Программа для робота, участвующего в олимпиаде "Робофест 2024 "в номнинации "Складские роботы"

Описание струкутры репозитория:
- [arduino/uart_i2c_conv/uart_i2c_conv.ino](arduino/uart_i2c_conv/uart_i2c_conv.ino) - файл содержит код для Arduino, 
который передаёт сообщения от EV3 к камере и обратно
- [ev3/ColorViewer.bp](ev3/ColorViewer.bp) - файл содержит код EV3, предназначенный для 
просмотра значений и калибровки датчиков цвета
- [ev3/robofest_24.bp](ev3/robofest_24.bp) - файл содержит основной цикл программы EV3
- [ev3/utils.bpi](ev3/utils.bpi) - файл содержит вспомогательные функции для программы 
EV3
- [ev3/Mods](ev3/Mods) - папка, содержащая множество вспомогательных функций 
для EV3 для работы с различными датчиками
- [ev3/Sounds](ev3/Sounds) - папка, содержащая звуки, используемые роботом
- [openmv/recognize_aruco_uart.py](openmv/recognize_aruco_uart.py) - файл, содержащий код обработки для 
камеры OpenMV
