SELECT * 
FROM properties_2017
JOIN predictions_2017 USING(id)
WHERE transactiondate BETWEEN '2017-05-01' AND '2017-09-01';

SELECT * 
FROM properties_2016
JOIN predictions_2016 USING(id)
WHERE transactiondate BETWEEN '2016-05-01' AND '2016-09-01';



-- 2016 Property Average Price by Zip Code
SELECT COUNT(regionidzip) AS zipcode_count, 
	regionidzip AS zipcode, 
	ROUND(AVG(taxvaluedollarcnt),0) AS zipcode_avg_price
FROM properties_2016
WHERE propertylandusetypeid = 261
GROUP BY regionidzip
ORDER BY AVG(taxvaluedollarcnt) DESC;

-- 2017 Single Unit Properties between May and Aug
SELECT bedroomcnt, bathroomcnt, 
	calculatedfinishedsquarefeet, 
	taxvaluedollarcnt, yearbuilt, 
	taxamount, fips, regionidzip 
FROM properties_2017
JOIN predictions_2017 USING(id)
WHERE propertylandusetypeid = 261
	AND transactiondate BETWEEN '2017-05-01' AND '2017-09-01';




-- properties 2016
SELECT COUNT(regionidzip), regionidzip, ROUND(AVG(taxvaluedollarcnt),0)
FROM properties_2016
WHERE propertylandusetypeid = 261
GROUP BY regionidzip
ORDER BY AVG(taxvaluedollarcnt) DESC;

SELECT COUNT(regionidzip), regionidzip, ROUND(AVG(taxvaluedollarcnt),0)
FROM properties_2017
JOIN properties_2016
	USING(parcelid)
WHERE propertylandusetypeid = 261
GROUP BY regionidzip
ORDER BY AVG(taxvaluedollarcnt) DESC;





/* SELECT COUNT(properties_2017.regionidzip), 
	properties_2017.regionidzip, 
	properties_2016.regionidzip, 
	ROUND(AVG(properties_2017.taxvaluedollarcnt),0)
FROM properties_2017
JOIN properties_2016 ON properties_2017.regionidzip = properties_2016.regionidzip
WHERE properties_2017.propertylandusetypeid = 261
GROUP BY properties_2017.regionidzip 
ORDER BY AVG(properties_2017.taxvaluedollarcnt) DESC; */

SELECT parcelid
FROM properties_2016
-- JOIN properties_2016
-- USING(parcelid);
