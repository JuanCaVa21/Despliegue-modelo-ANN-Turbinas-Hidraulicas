# Feature Store | Variables y Columnas

### Variables:
* **ID_Exp**: Id del experimento (numero incremental)               - Serial Primary Key
* **Torque**: Torque reflejado por el Torquimetro [Nm]              - Float
* **RPM**: Revoluciones por Minuto                                  - Float
* **Presion**: Presion sobre la turbina y el sistema [PSI]          - Float
* **Cauldal**: Caudal del agua arrojado por el sensor [l/s]         - Float
* **Potencia**: Potencia generada por la turbina en el exp [Watts]  - Float
* **Frecuencia**: Frencuencia del Exp [Hz]                          - Float

#### Parametros Opcionales:
* **n_of_alabes**: Numero de alabes que hay en la turbina               - Int
* **angulo_de_ataque**: Valor en grados del angulo de ataque del alabe  - Float
* **Date**: Fecha del Experimento                                       - Timestamp
* **Time**: Tiempo de duracion del experimento (empieza en 0.000s)      - Timestamp
* Se pueden agregar parametros tantos como sean necesario o deseados. Esto queda a dispocicion del semillero

