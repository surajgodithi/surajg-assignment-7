from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "boombmboombmbooofjdjsser"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots
    X = np.random.rand(N)
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, slope * X + intercept, color='red', label='Fitted Line')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_
        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    plot2_path = "static/plot2.png"
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    return (
        X, Y, slope, intercept, plot1_path, plot2_path, None, None, slopes, intercepts
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        X, Y, slope, intercept, plot1, plot2, _, _, slopes, intercepts = generate_data(
            N, mu, beta0, beta1, sigma2, S
        )

        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    N = session.get("N")
    if N is None:
        return "Error: 'N' value not found in session."
    N = int(N)

    S = int(session.get("S", 0))
    slope = float(session.get("slope", 0))
    intercept = float(session.get("intercept", 0))
    slopes = session.get("slopes", [])
    intercepts = session.get("intercepts", [])
    beta0 = float(session.get("beta0", 0))
    beta1 = float(session.get("beta1", 0))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    p_value = None
    if test_type == ">":
        p_value = sum(simulated_stats > observed_stat) / S
    elif test_type == "<":
        p_value = sum(simulated_stats < observed_stat) / S
    elif test_type == "!=":
        p_value = sum(abs(simulated_stats - hypothesized_value) >= abs(observed_stat - hypothesized_value)) / S

    # Debugging print statements
    print(f"Test Type: {test_type}")
    print(f"Observed Stat: {observed_stat}")
    print(f"Simulated Stats (sample): {simulated_stats[:10]}")
    print(f"Calculated p-value: {p_value}")

    if p_value is not None:
        if p_value <= 0.0001:
            fun_message = "Wow! You've encountered a very rare event with such a small p-value!"
        else:
            fun_message = None
    else:
        p_value = "Not calculated"
        fun_message = "Test type might be invalid or an error occurred."

    plot3_path = "static/plot3.png"
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_stats, bins=30, color='skyblue', alpha=0.7, label='Simulated Statistics')
    plt.axvline(observed_stat, color='red', linestyle='--', linewidth=2, label=f'Observed {parameter.capitalize()}: {observed_stat:.4f}')
    plt.axvline(hypothesized_value, color='blue', linestyle='-', linewidth=2, label=f'Hypothesized Value (H₀): {hypothesized_value}')
    plt.title(f'Hypothesis Test for {parameter.capitalize()}')
    plt.xlabel(f'{parameter.capitalize()} Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )
@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    print("Entered confidence_interval route")  # Debugging statement

    # Retrieve necessary session data
    N = session.get("N")
    if N is None:
        print("Error: 'N' value not found in session.")  # Debugging statement
        return "Error: 'N' value not found in session."
    N = int(N)

    S = int(session.get("S", 0))
    slopes = session.get("slopes", [])
    intercepts = session.get("intercepts", [])
    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    print(f"Received parameter: {parameter}, confidence_level: {confidence_level}")  # Debugging statement

    if parameter == "slope":
        estimates = np.array(slopes)
    else:
        estimates = np.array(intercepts)

    mean_estimate = np.mean(estimates)
    std_error = np.std(estimates) / np.sqrt(len(estimates))
    alpha = 1 - confidence_level / 100

    # Calculate Z-score approximation
    z_score = np.abs(np.percentile(np.random.normal(size=1000000), 100 * (1 - alpha / 2)))

    margin_of_error = z_score * std_error
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    true_param = session.get("beta1") if parameter == "slope" else session.get("beta0")
    includes_true = ci_lower <= true_param <= ci_upper

    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(estimates)), estimates, color='gray', alpha=0.5, label='Simulated Estimates')
    plt.axhline(mean_estimate, color='blue', linestyle='--', label=f'Mean Estimate: {mean_estimate:.4f}')
    plt.axhline(ci_lower, color='green', linestyle='--', label=f'Lower Bound: {ci_lower:.4f}')
    plt.axhline(ci_upper, color='red', linestyle='--', label=f'Upper Bound: {ci_upper:.4f}')
    plt.title(f'{confidence_level}% Confidence Interval for {parameter.capitalize()}')
    plt.xlabel('Simulation Index')
    plt.ylabel(f'{parameter.capitalize()} Estimate')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        confidence_level=confidence_level,
        
    )




if __name__ == "__main__":
    app.run(debug=True)
