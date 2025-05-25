# app.py

# Importar las librerías necesarias
from flask import Flask, request, jsonify
from flask_cors import CORS # Importar CORS para permitir peticiones desde tu frontend
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# Inicializar la aplicación Flask
app = Flask(__name__)
# Habilitar CORS para todas las rutas. Esto es CRUCIAL para que tu frontend (que corre en un puerto diferente)
# pueda acceder a tu backend sin problemas de seguridad del navegador.
CORS(app) 

# --- Cargar Modelos y Datos (Esta sección se ejecuta una única vez al iniciar la aplicación) ---
try:
    # Cargar el modelo optimizado de predicción de monto (RandomForestRegressor)
    pipeline_reg_monto_optimized = joblib.load('modelo_monto_recurrente_optimizado.pkl')
    # Cargar el modelo de clasificación de recurrencia
    modelo_recurrencia = joblib.load('modelo_recurrencia.pkl')
    
    # Cargar el DataFrame completo de transacciones desde el archivo Parquet
    # La ruta se asume relativa a la ubicación de app.py, subiendo un nivel y entrando en 'data'
    df_regression_full = pd.read_parquet(Path("./data/transacciones_para_modelo.parquet"))

    # Convertir la columna 'fecha' a tipo datetime, lo cual es esencial para cálculos temporales
    df_regression_full['fecha'] = pd.to_datetime(df_regression_full['fecha'])

    # --- Recalcular columnas necesarias para el modelo de regresión ---
    # Esta lógica debe ser EXACTAMENTE la misma que se usó durante el entrenamiento del modelo
    print("Calculando columnas necesarias para la app (monto_siguiente_recurrente, dias_hasta_siguiente_recurrencia)...")
    
    # Crear una clave única para agrupar transacciones por cliente y comercio
    df_regression_full['id_comercio'] = df_regression_full['id'].astype(str) + '_' + df_regression_full['comercio'].astype(str)
    
    # Ordenar las transacciones por esta clave y por fecha para asegurar el cálculo correcto de 'shift'
    df_regression_full = df_regression_full.sort_values(by=['id_comercio', 'fecha'])
    
    # Calcular el monto de la siguiente transacción para cada grupo (cliente-comercio)
    df_regression_full['monto_siguiente_recurrente'] = df_regression_full.groupby('id_comercio')['monto'].shift(-1)
    # Calcular la fecha de la siguiente transacción para cada grupo
    df_regression_full['fecha_siguiente'] = df_regression_full.groupby('id_comercio')['fecha'].shift(-1)
    
    # Si la transacción original NO fue recurrente ('es_recurrente' es False),
    # entonces el 'monto_siguiente_recurrente' y 'fecha_siguiente' no son relevantes para la recurrencia,
    # por lo que los establecemos a NaN.
    df_regression_full.loc[df_regression_full['es_recurrente'] == False, 'monto_siguiente_recurrente'] = np.nan
    df_regression_full.loc[df_regression_full['es_recurrente'] == False, 'fecha_siguiente'] = pd.NaT
    
    # Calcular los días hasta la siguiente recurrencia (diferencia en días entre la fecha actual y la siguiente)
    df_regression_full['dias_hasta_siguiente_recurrencia'] = (df_regression_full['fecha_siguiente'] - df_regression_full['fecha']).dt.days
    
    # Eliminar columnas temporales que ya no son necesarias
    df_regression_full.drop(columns=['id_comercio', 'fecha_siguiente'], errors='ignore', inplace=True)
    
    # Eliminar filas donde 'monto_siguiente_recurrente' es NaN.
    # Esto se hace porque estas transacciones no tienen una recurrencia válida para predecir.
    df_regression_full.dropna(subset=['monto_siguiente_recurrente'], inplace=True) 
    print(f"Filas en df_regression_full después de limpiar NaN: {len(df_regression_full)}")

    # --- Generar 'es_recurrente_pred' para TODO el df_regression_full ---
    print("Generando 'es_recurrente_pred' usando el modelo clasificador...")
    # Definir las características de entrada para el clasificador.
    # Excluir las columnas objetivo originales y el 'id' del cliente.
    features_for_classifier = [
        col for col in df_regression_full.columns
        if col not in ['id', 'es_recurrente', 'monto_siguiente_recurrente', 'dias_hasta_siguiente_recurrencia']
    ]
    
    # Verificar que todas las características necesarias para el clasificador están presentes
    missing_clf_features = [f for f in features_for_classifier if f not in df_regression_full.columns]
    if missing_clf_features:
        raise ValueError(f"Faltan las siguientes columnas para el clasificador: {missing_clf_features}")

    # Realizar la predicción de recurrencia y añadirla como una nueva columna
    df_regression_full['es_recurrente_pred'] = modelo_recurrencia.predict(df_regression_full[features_for_classifier])
    df_regression_full['es_recurrente_pred'] = df_regression_full['es_recurrente_pred'].astype(int)
    print("Columna 'es_recurrente_pred' generada para el DataFrame completo.")

except FileNotFoundError as e:
    print(f"Error al cargar modelos o datos: {e}. Asegúrate de que los archivos .pkl y .parquet estén en las rutas correctas.")
    # Salir de la aplicación si los archivos esenciales no se encuentran
    exit()
except Exception as e:
    print(f"Ocurrió un error inesperado durante la carga inicial o preprocesamiento: {e}")
    exit()

# Definir las características que el modelo de regresión espera como entrada.
# Esta lista debe coincidir EXACTAMENTE con las características usadas durante el entrenamiento del pipeline.
features_reg_model = [
    'fecha', 'comercio', 'giro_comercio', 'tipo_venta', 'monto',
    'edad_transaccion', 'genero', 'tipo_persona', 'antiguedad_cliente',
    'num_transacciones_previas_comercio', 'monto_promedio_comercio',
    'std_dias_entre_compras', 'diff_monto_promedio', 'monto_similar',
    'es_recurrente_pred', 'dias_desde_ultima_compra_comercio'
]


# --- Función auxiliar para predecir recurrencias futuras para un usuario específico ---
# Esta función simula las próximas transacciones recurrentes y sus montos.
def get_user_future_recurrences(user_id, num_months_forward=3):
    # Filtrar todas las transacciones del usuario dado
    user_transactions = df_regression_full[df_regression_full['id'] == user_id].copy()
    
    if user_transactions.empty:
        return [] # Si el usuario no tiene transacciones, no hay recurrencias que predecir

    # Ordenar las transacciones del usuario por fecha
    user_transactions = user_transactions.sort_values(by='fecha')
    
    # Para simplificar en el Datathon: tomamos la ÚLTIMA transacción para cada par único
    # de 'comercio' y 'tipo_venta' como base para proyectar futuras recurrencias.
    # En una aplicación real, esta lógica debería ser más sofisticada (ej. basada en frecuencia media).
    last_trans_per_comercio = user_transactions.drop_duplicates(subset=['comercio', 'tipo_venta'], keep='last')

    predicted_recurrences = []
    # Usamos la fecha de la última transacción conocida del usuario como el punto de inicio para proyectar.
    # Esto hace que las predicciones sean relativas a los datos más recientes del usuario.
    simulation_start_date = user_transactions['fecha'].max() 

    for index, row in last_trans_per_comercio.iterrows():
        # Solo consideramos proyectar transacciones que el clasificador predijo como recurrentes
        if row['es_recurrente_pred'] == 0:
            continue # Si no es recurrente, la ignoramos

        # Crear una "fila futura" basada en la transacción actual para pasar al modelo
        future_row = row.copy()
        
        # PROYECCIÓN SIMPLE PARA EL DATATHON: Asumir recurrencia mensual
        # Proyectamos para 'num_months_forward' meses hacia adelante.
        for m in range(1, num_months_forward + 1):
            projected_date = simulation_start_date + timedelta(days=30 * m) # Aproximación de un mes
            
            # Recalcular características dependientes de la fecha para la 'fecha_futura' proyectada.
            # Esto es CRÍTICO para que el pipeline reciba los datos como los espera.
            future_row['fecha'] = projected_date
            
            # Para 'dias_desde_ultima_compra_comercio', asumimos una recurrencia mensual (30 días).
            # En un modelo más avanzado, esto podría ser la frecuencia media del comercio/cliente.
            future_row['dias_desde_ultima_compra_comercio'] = 30 

            try:
                # Crear un DataFrame de 1 fila con las características para la predicción del monto
                df_for_prediction = pd.DataFrame([future_row[features_reg_model].to_dict()])
                
                # Realizar la predicción del monto usando el pipeline optimizado
                predicted_monto = pipeline_reg_monto_optimized.predict(df_for_prediction)[0]
                
                # Añadir la predicción a la lista de recurrencias futuras
                predicted_recurrences.append({
                    'comercio': future_row['comercio'],
                    'monto_predicho': round(float(predicted_monto), 2), # Redondear a 2 decimales
                    'fecha_predicha': projected_date.strftime('%Y-%m-%d'), # Formatear fecha
                    'tipo_recurrencia': 'predicha' # Etiqueta para el frontend
                })
            except KeyError as e:
                print(f"DEBUG: KeyError durante la predicción de monto: {e}. Revisa features_reg_model.")
            except Exception as e:
                print(f"DEBUG: Error general durante la predicción de monto: {e}")

    return predicted_recurrences

# --- Endpoint principal de la API Flask ---
# Esta ruta será llamada por tu frontend para obtener todo el estado financiero de un usuario.
@app.route('/predict_financial_status/<user_id>', methods=['GET'])
def predict_financial_status_api(user_id):
    # SIMULACIÓN DE DATOS INICIALES DEL USUARIO (ajusta estos valores según necesites para la demo)
    # En una aplicación real, estos datos vendrían de una base de datos de usuarios.
    
    # Obtener el saldo actual del usuario. Simplificación: basado en la última transacción del usuario.
    user_transactions = df_regression_full[df_regression_full['id'] == user_id].copy()
    if not user_transactions.empty:
        # Simular un saldo inicial multiplicando el monto de la última transacción
        saldo_actual = user_transactions['monto'].iloc[-1] * 5 
    else:
        saldo_actual = 500.00 # Saldo por defecto si el usuario no tiene transacciones históricas

    # Ingresos y gastos fijos simulados para la proyección
    ingresos_fijos_mensuales = 1500.00
    gastos_fijos_mensuales = 800.00
    meta_ahorro_mes = 200.00 # Meta de ahorro mensual
    ahorro_actual = 100.00 # Ahorro acumulado actual para el mes
    presupuesto_semanal_gastos = 300.00 # Presupuesto para gastos no recurrentes

    # Calcular próximas recurrencias y sus montos predichos para el usuario
    future_recurrences = get_user_future_recurrences(user_id)
    
    # --- 1. Lógica del "Balance Inteligente" ---
    today = datetime.now()
    # Sumar los montos de las recurrencias predichas que caen dentro de los próximos 30 días
    total_recurrencias_proximas_30d = sum(
        r['monto_predicho'] for r in future_recurrences
        if (datetime.strptime(r['fecha_predicha'], '%Y-%m-%d') - today).days <= 30 # Que estén en los próximos 30 días
        and (datetime.strptime(r['fecha_predicha'], '%Y-%m-%d') - today).days >= 0 # Y que no sean fechas pasadas
    )
    
    # Calcular el saldo proyectado al final del mes (aproximación)
    saldo_proyectado_mes = saldo_actual + ingresos_fijos_mensuales - gastos_fijos_mensuales - total_recurrencias_proximas_30d
    
    # Inicializar el diccionario de respuesta para el balance inteligente
    smart_balance = {
        "saldo_proyectado": round(saldo_proyectado_mes, 2),
        "alert": False,
        "message": "Todo bajo control.",
        "suggestion": ""
    }
    
    # Definir umbrales para las alertas del balance inteligente
    if saldo_proyectado_mes < 200: # Si el saldo proyectado es bajo, hay riesgo
        smart_balance["alert"] = True
        smart_balance["message"] = f"⚠️ ¡Alerta! Riesgo de no cubrir compromisos."
        smart_balance["suggestion"] = "💳 Considera microcréditos, aplazamientos, o revisa tus gastos."
    elif saldo_proyectado_mes > 1500: # Si hay mucha liquidez, sugerir inversión
        smart_balance["alert"] = False
        smart_balance["message"] = f"💡 ¡Excelente! Tienes liquidez."
        smart_balance["suggestion"] = "📈 Aprovecha para invertir o explorar promociones de tiendas afiliadas."
    
    # --- 2. Lógica del "Modo Conservar Recursos" ---
    conserve_mode = {
        "activated": False,
        "max_daily_spend": 0.0,
        "upcoming_payments_summary": []
    }
    
    # Calcular los días restantes hasta el fin del mes (simplificación)
    days_until_month_end = (pd.Period(today, 'M').end_time - today).days
    
    # Activar el modo conservar si es la última semana del mes Y el saldo actual es bajo
    if days_until_month_end <= 7 and saldo_actual < 800: 
        # Filtrar los pagos recurrentes que caen en lo que resta del mes
        upcoming_recurring_payments_end_of_month = [
            r for r in future_recurrences
            if (datetime.strptime(r['fecha_predicha'], '%Y-%m-%d') > today and (datetime.strptime(r['fecha_predicha'], '%Y-%m-%d') - today).days <= days_until_month_end)
        ]
        
        sum_upcoming_payments = sum(p['monto_predicho'] for p in upcoming_recurring_payments_end_of_month)
        
        # Si el saldo actual no es suficiente para cubrir los próximos pagos recurrentes
        if saldo_actual - sum_upcoming_payments < 0: 
            conserve_mode["activated"] = True
            if days_until_month_end > 0:
                 # Calcular el monto máximo que se puede gastar por día sin quedarse sin fondos
                 conserve_mode["max_daily_spend"] = max(0.0, (saldo_actual - sum_upcoming_payments) / days_until_month_end)
            else:
                 conserve_mode["max_daily_spend"] = 0.0 # Si ya es el último día del mes
            conserve_mode["upcoming_payments_summary"] = upcoming_recurring_payments_end_of_month
    
    # --- 3. Lógica del "Plan Financiero" ---
    financial_plan = {
        "savings_alert": False,
        "savings_message": "",
        "expenses_alert": False,
        "expenses_message": ""
    }
    
    # Ahorro: Proyección simple de ahorro al final del mes
    # Se asume que el ahorro proyectado es el saldo proyectado menos lo que ya tenías de saldo (si no es ahorro)
    projected_savings_at_month_end = saldo_proyectado_mes - (saldo_actual - ahorro_actual) 
    if projected_savings_at_month_end < meta_ahorro_mes:
        financial_plan["savings_alert"] = True
        financial_plan["savings_message"] = f"¡Alerta! No vas a llegar a tu meta de ahorro este mes (${meta_ahorro_mes:.2f}). Proyectado: ${projected_savings_at_month_end:.2f}."
        
    # Gastos: Para la demo, calculamos los gastos de la última semana de transacciones históricas del usuario
    last_known_transaction_date = user_transactions['fecha'].max() if not user_transactions.empty else today
    last_week_start = last_known_transaction_date - timedelta(days=7)
    expenses_last_week = user_transactions[
        (user_transactions['fecha'] >= last_week_start) & (user_transactions['fecha'] <= last_known_transaction_date)
    ]['monto'].sum()
    
    if expenses_last_week > presupuesto_semanal_gastos:
        financial_plan["expenses_alert"] = True
        financial_plan["expenses_message"] = f"¡Cuidado! Tus gastos de la última semana (${expenses_last_week:.2f}) superan el presupuesto (${presupuesto_semanal_gastos:.2f})."

    # Devolver todos los datos calculados en formato JSON al frontend
    return jsonify({
        "user_id": user_id,
        "saldo_actual": round(saldo_actual, 2),
        "ingresos_fijos_mensuales": ingresos_fijos_mensuales,
        "gastos_fijos_mensuales": gastos_fijos_mensuales,
        "meta_ahorro_mes": meta_ahorro_mes,
        "ahorro_actual": ahorro_actual,
        "presupuesto_semanal_gastos": presupuesto_semanal_gastos,
        "proximas_recurrencias": future_recurrences, # Lista de transacciones recurrentes predichas
        "balance_inteligente": smart_balance,
        "modo_conservar_recursos": conserve_mode,
        "plan_financiero": financial_plan
    })

# Bloque de ejecución principal para la aplicación Flask
if __name__ == '__main__':
    # Para ejecutar en un entorno de desarrollo, asegúrate de que tus modelos .pkl
    # estén en la misma carpeta que app.py, y la carpeta 'data' esté en el directorio superior.
    # O ajusta las rutas de carga según sea necesario.
    app.run(debug=True) # debug=True permite recarga automática y mensajes de error detallados
