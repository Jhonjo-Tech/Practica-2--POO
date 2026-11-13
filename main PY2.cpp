#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <cctype>

using namespace std;


// BLOQUE 1: CLASE LINEARREGRESSION (PROGRAMACION ORIENTADA A OBJETOS)

class LinearRegression {
private:
    vector<double> weights; // Coeficientes o pesos del modelo (w0, w1, ...)
    double bias;            // Termino independiente / intercepto (b)
    vector<double> meanX;   // Medias por columna para escalado
    vector<double> stdX;    // Desviaciones estandar por columna para escalado

public:
    // Constructor
    LinearRegression() : bias(0.0) {}

    // Destructor
    ~LinearRegression() {}

    // Getters de acceso requeridos
    vector<double> getWeights() const { return weights; }
    double getBias() const { return bias; }

    // 1. ESCALADO DE DATOS (Normalizacion Z-Score: (X - media) / desviacion)
    vector<vector<double> > data_scaling(const vector<vector<double> >& X) {
        int n = X.size();
        if (n == 0) return X;
        int num_features = X[0].size();

        meanX.assign(num_features, 0.0);
        stdX.assign(num_features, 0.0);

        // Paso A: Calcular la media de cada columna
        for (int j = 0; j < num_features; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += X[i][j];
            meanX[j] = sum / n;
        }

        // Paso B: Calcular la desviacion estandar de cada columna
        for (int j = 0; j < num_features; ++j) {
            double sq_sum = 0.0;
            for (int i = 0; i < n; ++i) {
                sq_sum += (X[i][j] - meanX[j]) * (X[i][j] - meanX[j]);
            }
            stdX[j] = sqrt(sq_sum / n);
            if (stdX[j] == 0) stdX[j] = 1.0; // Evitar division por cero
        }

        // Paso C: Aplicar la formula Z-Score
        vector<vector<double> > X_scaled = X;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < num_features; ++j) {
                X_scaled[i][j] = (X[i][j] - meanX[j]) / stdX[j];
            }
        }
        return X_scaled;
    }

    // 2. ENTRENAMIENTO
    void fit(const vector<vector<double> >& X, const vector<double>& y, double lr = 0.01, int epochs = 2500) {
        int n = X.size();
        if (n == 0) return;
        int num_features = X[0].size();

        weights.assign(num_features, 0.0);
        bias = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            vector<double> dw(num_features, 0.0);
            double db = 0.0;

            for (int i = 0; i < n; ++i) {
                double y_hat = bias;
                for (int j = 0; j < num_features; ++j) {
                    y_hat += weights[j] * X[i][j];
                }
                double error = y_hat - y[i];

                for (int j = 0; j < num_features; ++j) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }

            // Actualizacion de los pesos y el bias
            for (int j = 0; j < num_features; ++j) {
                weights[j] -= lr * (2.0 / n) * dw[j];
            }
            bias -= lr * (2.0 / n) * db;
        }
    }

    // 3. PREDICCION (predict - y_hat)
    vector<double> predict(const vector<vector<double> >& X) {
        int n = X.size();
        if (n == 0) return vector<double>();
        int num_features = X[0].size();
        vector<double> predictions(n);

        for (int i = 0; i < n; ++i) {
            double y_hat = bias;
            for (int j = 0; j < num_features; ++j) {
                y_hat += weights[j] * X[i][j];
            }
            predictions[i] = y_hat;
        }
        return predictions;
    }

    // 4. EVALUACION (score - Coeficiente de Determinacion R^2)
    double score(const vector<double>& y_true, const vector<double>& y_pred) {
        double mean_y = 0.0;
        int n = y_true.size();
        if (n == 0) return 0.0;
        for (int i = 0; i < n; ++i) mean_y += y_true[i];
        mean_y /= n;

        double ss_tot = 0.0, ss_res = 0.0;
        for (int i = 0; i < n; ++i) {
            ss_tot += (y_true[i] - mean_y) * (y_true[i] - mean_y);
            ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        }
        if (ss_tot == 0.0) return 0.0;
        return 1.0 - (ss_res / ss_tot);
    }

    // Metrica complementaria: Error Cuadratico Medio (MSE)
    double get_mse(const vector<double>& y_true, const vector<double>& y_pred) {
        double mse = 0.0;
        int n = y_true.size();
        if (n == 0) return 0.0;
        for (int i = 0; i < n; ++i) {
            mse += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        }
        return mse / n;
    }
};


// BLOQUE 2: LECTURA DE ARCHIVOS 

bool load_csv(const string& filename, vector<vector<double> >& X, vector<double>& y) {
    ifstream file(filename.c_str());
    if (!file.is_open()) {
        cerr << " [!] Error: No se pudo abrir el archivo " << filename << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        // Limpiar retorno de carro \r si existe (formato Windows/CRLF)
        if (!line.empty() && line[line.size() - 1] == '\r') {
            line.erase(line.size() - 1);
        }
        if (line.empty()) continue;

        stringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ',')) {
            if (!cell.empty() && cell[cell.size() - 1] == '\r') {
                cell.erase(cell.size() - 1);
            }
            if (cell.empty()) continue;

            // Verificar si contiene caracteres numericos
            bool is_num = false;
            for (size_t k = 0; k < cell.size(); ++k) {
                if (isdigit(cell[k]) || cell[k] == '-' || cell[k] == '.') {
                    is_num = true;
                    break;
                }
            }

            if (is_num) {
                double val = atof(cell.c_str());
                row.push_back(val);
            }
        }

        if (!row.empty()) {
            y.push_back(row.back()); // La ultima columna es la variable objetivo (y)
            row.pop_back();          // Las primeras columnas son las variables de entrada (X)
            X.push_back(row);
        }
    }
    file.close();
    return true;
}

// Objeto reservado en Memoria Global (Fuera de main)
LinearRegression globalModel;

// FUNCION PRINCIPAL MAIN

int main() {
    cout << "=========================================================" << endl;
    cout << "   PRACTICA 2: REGRESION LINEAL Y MEMORIA RAM (C++)" << endl;
    cout << "=========================================================\n" << endl;

    // ------------------------------------------------------------------------
    // PRUEBA 1: DATASET HELADOS (Regresion Simple)
    // ------------------------------------------------------------------------
    vector<vector<double> > X_ice;
    vector<double> y_ice;
    if (load_csv("Ice_cream_selling_data.csv", X_ice, y_ice)) {
        cout << ">>> 1. DATASET: Ventas de Helado (Regresion Simple) <<<" << endl;
        LinearRegression model_ice;
        vector<vector<double> > X_scaled = model_ice.data_scaling(X_ice);
        model_ice.fit(X_scaled, y_ice);
        vector<double> y_pred = model_ice.predict(X_scaled);

        cout << "   Weights: ";
        vector<double> w = model_ice.getWeights();
        for (size_t i = 0; i < w.size(); ++i) cout << w[i] << " ";
        cout << "\n   Bias: " << model_ice.getBias() << endl;
        
        cout << "   Muestra de predicciones (y_hat vs y_real):" << endl;
        for (size_t i = 0; i < 3 && i < y_pred.size(); ++i) {
            cout << "     Muestra [" << i+1 << "] -> Predicho (y_hat): " 
                 << y_pred[i] << " | Real: " << y_ice[i] << endl;
        }

        cout << "   MSE Error: " << model_ice.get_mse(y_ice, y_pred) << endl;
        cout << "   R^2 Score: " << model_ice.score(y_ice, y_pred) << " (Nota: relacion no lineal, parabolica)" << endl << endl;
    }

    // PRUEBA 2: DATASET ESTUDIANTES (Regresion Multiple)

    vector<vector<double> > X_stud;
    vector<double> y_stud;
    if (load_csv("student_exam_scores.csv", X_stud, y_stud)) {
        cout << ">>> 2. DATASET: Notas de Estudiantes (Regresion Multiple) <<<" << endl;
        LinearRegression model_stud;
        vector<vector<double> > X_scaled = model_stud.data_scaling(X_stud);
        model_stud.fit(X_scaled, y_stud);
        vector<double> y_pred = model_stud.predict(X_scaled);

        cout << "   Weights (4 variables): ";
        vector<double> w = model_stud.getWeights();
        for (size_t i = 0; i < w.size(); ++i) cout << w[i] << " ";
        cout << "\n   Bias: " << model_stud.getBias() << endl;
        
        cout << "   Muestra de predicciones (y_hat vs y_real):" << endl;
        for (size_t i = 0; i < 3 && i < y_pred.size(); ++i) {
            cout << "     Estudiante [" << i+1 << "] -> Predicho: " 
                 << y_pred[i] << " | Real: " << y_stud[i] << endl;
        }
        cout << "   MSE Error: " << model_stud.get_mse(y_stud, y_pred) << endl;
        cout << "   R^2 Score Modelo: " << model_stud.score(y_stud, y_pred) << endl << endl;
    }


    // PRUEBA 3: ANALISIS DE MEMORIA RAM (Global, Stack, Heap y Arreglo Dinamico)
 
    cout << "=========================================================" << endl;
    cout << "            ANALISIS DE MEMORIA EN C++" << endl;
    cout << "=========================================================" << endl;

    // Memoria Local (Stack)
    LinearRegression stackModel;

    // Memoria Dinamica (Heap)
    LinearRegression* heapModel = new LinearRegression();

    // Arreglo Dinamico de 3 Objetos en Memoria Dinamica (Heap)
    int n_objetos = 3;
    LinearRegression* dynamicArray = new LinearRegression[n_objetos];

    cout << "1. Objeto en Memoria Global:" << endl;
    cout << "   -> Direccion (&): " << &globalModel << endl;
    cout << "   -> Tamano (sizeof): " << sizeof(globalModel) << " bytes" << endl << endl;

    cout << "2. Objeto en Memoria Local (Stack):" << endl;
    cout << "   -> Direccion (&): " << &stackModel << endl;
    cout << "   -> Tamano (sizeof): " << sizeof(stackModel) << " bytes" << endl << endl;

    cout << "3. Objeto Unico en Memoria Dinamica (Heap):" << endl;
    cout << "   -> Puntero en el Stack (&): " << &heapModel << endl;
    cout << "   -> Objeto real en el Heap:  " << heapModel << endl;
    cout << "   -> Tamano del objeto:      " << sizeof(*heapModel) << " bytes" << endl << endl;

    cout << "4. Arreglo Dinamico de " << n_objetos << " Objetos en Heap:" << endl;
    cout << "   -> Direccion base en Heap:  " << dynamicArray << endl;
    cout << "   -> Tamano del puntero:     " << sizeof(dynamicArray) << " bytes" << endl;
    cout << "   -> Tamano total del arreglo en Heap: " << sizeof(LinearRegression) * n_objetos << " bytes" << endl;

    // Liberacion explicita de memoria dinamica
    delete heapModel;
    delete[] dynamicArray;

    return 0;
}
