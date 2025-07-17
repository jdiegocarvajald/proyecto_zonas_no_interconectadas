CREATE OR REPLACE VIEW energias.servicios_detalle AS
SELECT 
	d.id_departamento AS "Código Departamento",
	d.departamento AS "Departamento",
	m.id_municipio AS "Código Municipio",
	m.municipio AS "Municipio",
	c.id_centro_poblado AS "Código Centro Poblado",
	c.centro_poblado AS "Centro Poblado",
	e.energia_activa AS "Energía Activa [kWh]",
	e.energia_reactiva AS "Energía Reactiva [kVArh]",
	e.potencia_maxima AS "Potencia Máxima [kW]",
	e.fecha_demanda_maxima AS "Fecha Demanda Máxima",
	e.promedio_diario_horas AS "Promedio Diario [h]"
	--e.energia_activa / SQRT(POWER(e.energia_activa, 2) + POWER(e.energia_reactiva, 2)) AS "Factor de Potencia"
FROM energias.servicios_centros_poblados e
LEFT JOIN divapola.centros_poblados c ON e.id_centro_poblado = c.id_centro_poblado
LEFT JOIN divapola.municipios m ON c.id_municipio = m.id_municipio
LEFT JOIN divapola.departamentos d ON m.id_departamento = d.id_departamento;