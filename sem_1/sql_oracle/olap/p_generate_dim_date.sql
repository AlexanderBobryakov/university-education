CREATE OR REPLACE PROCEDURE P_GENERATE_DIM_DATE 
(
  DATE_FROM IN DATE 
, DATE_TO IN DATE 
) AS 
d_date DATE := DATE_FROM;
BEGIN
  LOOP
    INSERT INTO DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")
      values (SEQ_DIM_DATE.NEXTVAL, TO_CHAR(d_date, 'DD.MM.YYYY'), 1);
    if (EXTRACT(DAY FROM d_date) = 1) then
      INSERT INTO DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")
      values (SEQ_DIM_DATE.NEXTVAL, TO_CHAR(d_date, 'MM.YYYY'), 2);
      
      if (EXTRACT(MONTH FROM d_date) = 1) then
        INSERT INTO DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")
          values (SEQ_DIM_DATE.NEXTVAL, TO_CHAR(d_date, 'YYYY'), 3);
      end if;
    end if;
    d_date := d_date + 1;
    EXIT WHEN d_DATE >= DATE_TO;
  END LOOP;
    
END P_GENERATE_DIM_DATE;
