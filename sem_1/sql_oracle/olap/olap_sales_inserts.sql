create sequence SEQ_DIM_COUNTRY;

select seq_dim_country.nextval from dual;

insert into DIM_COUNTRY(COUNTRY_ID, COUNTRY_NAME)
select SEQ_DIM_COUNTRY.nextval, country from 
(select distinct country from city);
commit;

select * from DIM_COUNTRY;

insert into DIM_CITY(CITY_ID, CITY_NAME, COUNTRY_ID)
select CITY_ID, CITY_NAME, (select COUNTRY_ID from DIM_COUNTRY where COUNTRY_NAME = CITY.COUNTRY)
from CITY;
commit;

select * from DIM_CITY;

insert into DIM_OFFICE(OFFICE_ID, OFFICE_NAME, CITY_ID)
select OFFICE_ID, OFFICE_NAME, CITY_ID
from OFFICE;
commit;

select * from DIM_OFFICE;

insert into DIM_MANAGER(MANAGER_ID, LAST_NAME, FIRST_NAME)
select MANAGER_ID, MANAGER_LAST_NAME, MANAGER_FIRST_NAME
from MANAGER;
commit;

insert into DIM_PRODUCT(PRODUCT_ID, PRODUCT_NAME)
select PRODUCT_ID, PRODUCT_NAME
from PRODUCT;

commit;

select TO_CHAR(min(ORDER_DATE), 'DD.MM.YYYY'), max(ORDER_DATE)
from SALES_ORDER;

select distinct ORDER_DATE from SALES_ORDER;

create sequence SEQ_DIM_DATE;

begin
  P_GENERATE_DIM_DATE(TO_DATE('01.01.2000', 'DD.MM.YYYY'), TO_DATE('01.01.2017', 'DD.MM.YYYY'));
end;

rollback;

select * from DIM_DATE where "LEVEL"=3;

commit;


create table STG_FACT_SALES as
select sol.ORDER_LINE_ID,
         sol.product_id,
         so.SALES_ORDER_ID,
         so.manager_id,
         m.MANAGER_FIRST_NAME,
         m.MANAGER_LAST_NAME,
         m.office_id,
         o.OFFICE_NAME,
         o.city_id,
         c.CITY_NAME,
         c.COUNTRY,
         c.REGION,
         sol.product_qty,
         sol.product_price,
         so.order_date
from SALES_ORDER_LINE SOL
inner join SALES_ORDER SO on (SOL.SALES_ORDER_ID = SO.SALES_ORDER_ID)
inner join MANAGER M on (M.MANAGER_ID = SO.MANAGER_ID)
inner join OFFICE O on (O.OFFICE_ID = M.OFFICE_ID)
inner join CITY C on (O.CITY_ID = C.CITY_ID);

select * from STG_FACT_SALES;

insert into FACT_SALES(
SALE_ID,
PRODUCT_ID,
MANAGER_ID,
COUNTRY_ID,
CITY_ID,
OFFICE_ID,
DATE_ID,
SALE_AMOUNT,
SALE_PRICE,
SALE_QTY
)
select 
ORDER_LINE_ID,
PRODUCT_ID,
MANAGER_ID,
(select COUNTRY_ID from DIM_COUNTRY C where C.COUNTRY_NAME = STG_FACT_SALES.COUNTRY),
CITY_ID,
OFFICE_ID,
(select DATE_ID from DIM_DATE D where D.DATE_NAME = TO_CHAR(STG_FACT_SALES.order_date, 'DD.MM.YYYY') and "LEVEL" = 1),
product_PRICE * product_qty,
product_PRICE,
product_qty
from STG_FACT_SALES;

commit;

select * from FACT_SALES;

select PRODUCT_ID, NULL, NULL, NULL, NULL, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by PRODUCT_ID
  union all
select NULL, MANAGER_ID, NULL, NULL, NULL, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by MANAGER_ID
  union all
select PRODUCT_ID, MANAGER_ID, NULL, NULL, NULL, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by PRODUCT_ID, MANAGER_ID
  union all
select NULL, NULL, NULL, NULL, NULL, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES;

create table FACT_SALES_AGG as select * from FACT_SALES where ROWNUM < 1;

insert into FACT_SALES_AGG(
SALE_ID,
PRODUCT_ID,
MANAGER_ID,
DATE_ID,
OFFICE_ID,
CITY_ID,
COUNTRY_ID,
SALE_AMOUNT,
SALE_PRICE,
SALE_QTY
)
select NULL, PRODUCT_ID, MANAGER_ID, DATE_ID, OFFICE_ID, CITY_ID, COUNTRY_ID, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by CUBE(PRODUCT_ID, MANAGER_ID, DATE_ID), ROLLUP(COUNTRY_ID, CITY_ID, OFFICE_ID);

select * from FACT_SALES_AGG;

create materialized view MV_FACT_SALES_AGG as
select PRODUCT_ID, MANAGER_ID, DATE_ID, OFFICE_ID, CITY_ID, COUNTRY_ID, SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by CUBE(PRODUCT_ID, MANAGER_ID, DATE_ID), ROLLUP(COUNTRY_ID, CITY_ID, OFFICE_ID);

select * from MV_FACT_SALES_AGG;

begin
   dbms_mview.refresh('MV_FACT_SALES_AGG');
end;
