#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

using namespace std;

struct Result {
    double x1, x2;
    double f_val;
    int iterations;
};

double f(double x1, double x2) {
    return 3.0 * x1 * x1 + 0.6 * x1 * x2 + 3.0 * x2 * x2;
}

vector<double> grad_f(double x1, double x2) {
    double df1 = 6.0 * x1 + 0.6 * x2;
    double df2 = 0.6 * x1 + 6.0 * x2;
    return { df1, df2 };
}

vector<vector<double>> hessian() {
    return { {6.0, 0.6},
             {0.6, 6.0} };
}

double det2x2(const vector<vector<double>>& M) {
    return M[0][0] * M[1][1] - M[0][1] * M[1][0];
}

vector<vector<double>> inv2x2(const vector<vector<double>>& M) {
    double det = det2x2(M);
    if (fabs(det) < 1e-12)
        throw runtime_error("Matrix is singular");
    return { { M[1][1] / det, -M[0][1] / det },
             {-M[1][0] / det,  M[0][0] / det } };
}

bool is_positive_definite(const vector<vector<double>>& M) {
    return (M[0][0] > 0) && (det2x2(M) > 0);
}

vector<double> mat_vec_mult(const vector<vector<double>>& A, const vector<double>& v) {
    return { A[0][0] * v[0] + A[0][1] * v[1],
             A[1][0] * v[0] + A[1][1] * v[1] };
}

double norm(const vector<double>& v) {
    return sqrt(v[0] * v[0] + v[1] * v[1]);
}

Result newton_method(double x1_0, double x2_0, double eps1, double eps2, int M) {
    double x1 = x1_0, x2 = x2_0;
    int k = 0;

    double prev_x1 = x1, prev_x2 = x2;
    double prev_f = f(x1, x2);
    bool first_iter = true;

    cout << "\n=== Метод Ньютона ===\n";
    cout << "Итерация 0: x = (" << x1 << ", " << x2 << "), f = " << prev_f << "\n";

    while (true) {
        auto grad = grad_f(x1, x2);
        double grad_norm = norm(grad);

        if (grad_norm <= eps1) {
            cout << "Остановка: норма градиента (" << grad_norm << ") <= eps1 (" << eps1 << ")\n";
            break;
        }
        if (k >= M) {
            cout << "Остановка: достигнуто предельное число итераций M = " << M << "\n";
            break;
        }

        auto H = hessian();

        vector<vector<double>> H_inv;
        bool use_newton = true;
        try {
            H_inv = inv2x2(H);
        }
        catch (const runtime_error&) {
            use_newton = false;
        }
        if (use_newton && !is_positive_definite(H)) {
            use_newton = false;
        }

        vector<double> d;
        if (use_newton) {
            auto H_inv_grad = mat_vec_mult(H_inv, grad);
            d = { -H_inv_grad[0], -H_inv_grad[1] };
            cout << "Итерация " << k + 1 << ": направление Ньютона\n";
        }
        else {
            d = { -grad[0], -grad[1] };
            cout << "Итерация " << k + 1 << ": направление антиградиента (Гессиан не полож. опред.)\n";
        }

        double t = 1.0;
        if (!use_newton) {
            while (true) {
                double new_x1 = x1 + t * d[0];
                double new_x2 = x2 + t * d[1];
                if (f(new_x1, new_x2) < f(x1, x2)) break;
                t /= 2.0;
                if (t < 1e-12) break;
            }
            cout << "  Подобран шаг t = " << t << "\n";
        }

        double new_x1 = x1 + t * d[0];
        double new_x2 = x2 + t * d[1];
        double new_f = f(new_x1, new_x2);

        if (!first_iter) {
            double dx = sqrt(pow(new_x1 - prev_x1, 2) + pow(new_x2 - prev_x2, 2));
            double df = fabs(new_f - prev_f);
            cout << "  dx = " << dx << ", df = " << df << "\n";
            if (dx < eps2 && df < eps2) {
                cout << "Остановка: дважды подряд выполнены условия по eps2\n";
                x1 = new_x1; x2 = new_x2;
                k++;
                break;
            }
        }

        prev_x1 = x1; prev_x2 = x2;
        prev_f = f(x1, x2);
        x1 = new_x1; x2 = new_x2;
        k++;
        first_iter = false;

        cout << "  x = (" << x1 << ", " << x2 << "), f = " << new_f << ", ||grad|| = " << norm(grad_f(x1, x2)) << "\n";
    }

    return { x1, x2, f(x1, x2), k };
}

Result marquardt_method(double x1_0, double x2_0, double eps1, int M, double mu0) {
    double x1 = x1_0, x2 = x2_0;
    double mu = mu0;
    int k = 0;

    cout << "\n=== Метод Марквардта ===\n";
    cout << "Итерация 0: x = (" << x1 << ", " << x2 << "), f = " << f(x1, x2) << ", mu = " << mu << "\n";

    while (true) {
        auto grad = grad_f(x1, x2);
        double grad_norm = norm(grad);

        if (grad_norm <= eps1) {
            cout << "Остановка: норма градиента (" << grad_norm << ") <= eps1 (" << eps1 << ")\n";
            break;
        }
        if (k >= M) {
            cout << "Остановка: достигнуто предельное число итераций M = " << M << "\n";
            break;
        }

        auto H = hessian();

        vector<vector<double>> H_reg = {
            {H[0][0] + mu, H[0][1]},
            {H[1][0], H[1][1] + mu}
        };

        vector<vector<double>> H_reg_inv;
        try {
            H_reg_inv = inv2x2(H_reg);
        }
        catch (const runtime_error&) {
            mu *= 2.0;
            cout << "Итерация " << k + 1 << ": матрица вырождена, mu увеличен до " << mu << "\n";
            continue;
        }

        auto H_inv_grad = mat_vec_mult(H_reg_inv, grad);
        vector<double> d = { -H_inv_grad[0], -H_inv_grad[1] };

        double x1_new = x1 + d[0];
        double x2_new = x2 + d[1];
        double f_old = f(x1, x2);
        double f_new = f(x1_new, x2_new);

        cout << "Итерация " << k + 1 << ": mu = " << mu << ", направление d = (" << d[0] << ", " << d[1] << ")\n";
        cout << "  x_new = (" << x1_new << ", " << x2_new << "), f_new = " << f_new << "\n";

        if (f_new < f_old) {
            mu /= 2.0;
            x1 = x1_new;
            x2 = x2_new;
            k++;
            cout << "  Шаг удачный: mu уменьшен до " << mu << ", точка принята\n";
            cout << "  x = (" << x1 << ", " << x2 << "), f = " << f_new << ", ||grad|| = " << norm(grad_f(x1, x2)) << "\n";
        }
        else {
            mu *= 2.0;
            cout << "  Шаг неудачный: mu увеличен до " << mu << ", точка не изменяется\n";
        }
    }

    return { x1, x2, f(x1, x2), k };
}

int main() {
    setlocale(LC_ALL, "RUS");

    double x1_0 = 1.5, x2_0 = 0.5;
    double eps1 = 0.15;
    double eps2 = 0.2;
    int M = 10;
    double mu0 = 100.0;

    cout << "Функция: f(x) = 3*x1^2 + 0.6*x1*x2 + 3*x2^2\n";
    cout << "Начальная точка: (" << x1_0 << ", " << x2_0 << ")\n";
    cout << "eps1 = " << eps1 << ", eps2 = " << eps2 << ", M = " << M << ", mu0 = " << mu0 << "\n";

    Result res_newton = newton_method(x1_0, x2_0, eps1, eps2, M);
    Result res_marquardt = marquardt_method(x1_0, x2_0, eps1, M, mu0);

    cout << "\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===\n";
    cout << "Метод Ньютона:\n";
    cout << "  x* = (" << res_newton.x1 << ", " << res_newton.x2 << ")\n";
    cout << "  f(x*) = " << res_newton.f_val << "\n";
    cout << "  Итераций: " << res_newton.iterations << "\n\n";

    cout << "Метод Марквардта:\n";
    cout << "  x* = (" << res_marquardt.x1 << ", " << res_marquardt.x2 << ")\n";
    cout << "  f(x*) = " << res_marquardt.f_val << "\n";
    cout << "  Итераций: " << res_marquardt.iterations << "\n";
}