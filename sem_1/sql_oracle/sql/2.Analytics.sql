-- 1. Каждый месяц компания выдает премию в размере 5% от суммы продаж менеджеру,
-- который за предыдущие 3 месяца продал товаров на самую большую сумму
-- Выведите месяц, manager_id, manager_first_name, manager_last_name,
-- премию за период с января по декабрь 2014 года
select MONTH,
       manager_id,
       manager_first_name,
       manager_last_name,
       PERIOD_SALE_AMOUNT,
       0.05 * PERIOD_SALE_AMOUNT BONUS_AMOUNT
from (select MONTH,
             manager_id,
             manager_first_name,
             manager_last_name,
             PERIOD_SALE_AMOUNT,
             MAX(PERIOD_SALE_AMOUNT) over (partition by MONTH) MAX_MONTH_SALE_AMOUNT
      from (select MONTH,
                   manager_id,
                   manager_first_name,
                   manager_last_name,
                   SUM(sale_amount) OVER (PARTITION BY manager_id order by MONTH RANGE 3 PRECEDING) PERIOD_SALE_AMOUNT
            from (select TRUNC(MONTHS_BETWEEN(order_date, '01.01.2014')) + 1 MONTH,
                         manager_id,
                         manager_first_name,
                         manager_last_name,
                         sum(sale_amount)                                    SALE_AMOUNT
                  from v_fact_sale
                  where order_date between '01.10.2013' and '31.12.2014'
                        --
                  group by (
                            TRUNC(MONTHS_BETWEEN(order_date, '01.01.2014')) + 1,
                            manager_id,
                            manager_first_name,
                            manager_last_name)) STEP1) STEP2) STEP3
     --
where PERIOD_SALE_AMOUNT = MAX_MONTH_SALE_AMOUNT
  and MONTH > 0;

-- 2. Компания хочет оптимизировать количество офисов, проанализировав относительные объемы продаж по офисам
-- в течение периода с 2013-2014 гг.
-- Выведите год, office_id, city_name, country, относительный объем продаж за текущий год
-- Офисы, которые демонстрируют наименьший относительной объем в течение двух лет скорее всего будут закрыты
select *
from (
         -- добавляем долю продаж офиса среди всех офисов по соответствующему году
         select extract(YEAR from order_date)                                             as year,
                office_id,
                city_name,
                country,
                RATIO_TO_REPORT(volume) over (PARTITION BY extract(YEAR from order_date)) as percent_by_year
         from (
                  -- добавляем объем продаж для офиса по текущему году
                  SELECT v.*,
                         sum(sale_amount) over (partition by OFFICE_ID, extract(YEAR from order_date)) as volume
                  FROM v_fact_sale v
                  where extract(YEAR from order_date) between 2013 and 2014
                    and office_id is not null)
         order by OFFICE_ID, year)
group by year, OFFICE_ID, CITY_NAME, COUNTRY, percent_by_year
order by OFFICE_ID;

-- 3)  Для планирования закупок, компанию оценивает динамику роста продаж по товарам.
-- Динамика оценивается как отношение объема продаж в текущем месяце к предыдущему.
-- Выведите товары, которые демонстрировали наиболее высокие темпы роста продаж в течение первого полугодия 2014 года.
select PRODUCT_ID,
       max(score) max_product_score
from (
         -- получаем в score все динамики роста для кадого товара каждого месяца
         select PRODUCT_ID,
                nvl(
                            sum(SALE_AMOUNT)
                            / lag(sum(SALE_AMOUNT)) over (partition by PRODUCT_ID order by extract(MONTH from ORDER_DATE)),
                            0) score,
                extract(MONTH from ORDER_DATE)
         from V_FACT_SALE v
         where ORDER_DATE between '01.12.2013' and '30.06.2014'
         group by PRODUCT_ID, extract(MONTH from ORDER_DATE)
         order by PRODUCT_ID, extract(MONTH from ORDER_DATE)) t
group by PRODUCT_ID
order by max_product_score desc;

-- 4) Напишите запрос, который выводит отчет о прибыли компании за 2014 год: помесячно и поквартально.
-- Отчет включает сумму прибыли за период и накопительную сумму прибыли с начала года по текущий период.
select quarter,
       month,
       sale_amount                                                                               as "помесячно",
       sum(sale_amount) over (order by month rows between unbounded preceding and current row)   as "помес -сНачала",
       sum(sale_amount) over (partition by quarter)                                              as "поквартально",
       sum(sale_amount) over (order by quarter range between unbounded preceding and current row) as "покварт -сНачала",
from (
         -- сумма продаж по каждому месяцу + пишем квартал
         select to_char(ORDER_DATE, 'MM') month,
                to_char(ORDER_DATE, 'Q')  quarter,
                sum(SALE_AMOUNT)          sale_amount
         from v_fact_sale
         where ORDER_DATE between '01.01.2014' and '31.12.2014'
         group by to_char(ORDER_DATE, 'MM'), to_char(ORDER_DATE, 'Q')
     )
order by month;

select month,
       quarter,
       sale_amount                                                                                as "помесячно",
       sum(sale_amount) over (order by month rows between unbounded preceding and current row)   as "помес -сНачала",
       sum(sale_amount) over (partition by quarter)                                              as "поквартально",
       sum(sale_amount) over (order by quarter rows between unbounded preceding and current row) as "покварт -сНачала"
from (
         select to_char(ORDER_DATE, 'MM') month,
                to_char(ORDER_DATE, 'Q')  quarter,
                sum(SALE_AMOUNT)          sale_amount
         from v_fact_sale
         where ORDER_DATE between '01.01.2014' and '31.12.2014'
         group by to_char(ORDER_DATE, 'MM'), to_char(ORDER_DATE, 'Q'))
order by month;


--  5. Найдите вклад в общую прибыль за 2014 год 10% наиболее дорогих товаров и 10% наиболее дешевых товаров.
--     Выведите product_id, product_name, total_sale_amount, percent
select product_id,
       product_name,
       total_PRODUCT_PRICE,
       round(percent * 100, 4) persent
from (
         select product_id,
                product_name,
                sum(PRODUCT_PRICE)                             total_PRODUCT_PRICE,
                cume_dist() over (order by sum(PRODUCT_PRICE)) dist,   -- для определения дорогих/дешевых
                RATIO_TO_REPORT(sum(PRODUCT_PRICE)) OVER () AS percent -- вклад в суммарные продажи
         from v_fact_sale
         where ORDER_DATE between '01.01.2014' and '31.12.2014'
         group by product_id, PRODUCT_NAME
         order by PRODUCT_ID
     )
where dist >= 0.9
   or dist <= 0.1;

-- 6. Компания хочет премировать трех наиболее продуктивных (по объему продаж, конечно) менеджеров в каждой стране в 2014 году.
-- Выведите country, <список manager_last_name manager_first_name, разделенный запятыми> которым будет выплачена премия
select COUNTRY,
       listagg(MANAGER_LAST_NAME || ' ' || MANAGER_FIRST_NAME, ', ') within group (order by MANAGER_LAST_NAME )
from (
         -- Добавили номера строк в группах из которых отсекаем 3 лучших по сумме продаж
         select i.*,
                row_number() over (partition by COUNTRY order by sum desc) group_row
         from (
                  -- суммарные продажи по менеджерам в каждой стране
                  select COUNTRY,
                         MANAGER_ID,
                         MANAGER_FIRST_NAME,
                         MANAGER_LAST_NAME,
                         sum(SALE_AMOUNT) over (partition by COUNTRY, MANAGER_ID) sum
                  from V_FACT_SALE
                  where extract(YEAR from ORDER_DATE) = 2014
                  order by COUNTRY, sum
              ) i
         group by COUNTRY, MANAGER_ID, MANAGER_FIRST_NAME, MANAGER_LAST_NAME, sum
     )
where group_row < 5
group by COUNTRY;

--  7. Выведите самый дешевый и самый дорогой товар, проданный за каждый месяц в течение 2014 года.
-- cheapest_product_id, cheapest_product_name, expensive_product_id, expensive_product_name, month, cheapest_price, expensive_price
select distinct month,
                first_value(PRODUCT_ID) over (partition by month order by PRODUCT_PRICE)    as cheapest_product_id,
                first_value(PRODUCT_NAME) over (partition by month order by PRODUCT_PRICE)  as cheapest_product_name,
                first_value(PRODUCT_PRICE) over (partition by month order by PRODUCT_PRICE) as cheapest_price,
                last_value(PRODUCT_ID)
                           over (partition by month order by PRODUCT_PRICE rows between unbounded preceding and unbounded following)
                                                                                            as expensive_product_id,
                last_value(PRODUCT_NAME)
                           over (partition by month order by PRODUCT_PRICE rows between unbounded preceding and unbounded following)
                                                                                            as expensive_product_name,
                last_value(PRODUCT_PRICE)
                           over (partition by month order by PRODUCT_PRICE rows between unbounded preceding and unbounded following)
                                                                                            as expensive_price
from (
         -- все цены товаров по каждому месяцу
         select extract(MONTH from ORDER_DATE) month, PRODUCT_ID, PRODUCT_NAME, PRODUCT_PRICE
         from V_FACT_SALE
         where ORDER_DATE between '01.01.2014' and '31.12.2014'
         order by extract(MONTH from ORDER_DATE), PRODUCT_PRICE
     )
order by month;

-- 8. Менеджер получает оклад в 30 000 + 5% от суммы своих продаж в месяц. Средняя наценка стоимости товара - 10%
-- Посчитайте прибыль предприятия за 2014 год по месяцам (сумма продаж - (исходная стоимость товаров + зарплата))
-- month, sales_amount, salary_amount, profit_amount
select month,
       sum(manager_sales)                                              as sales_amount,
       sum(manager_salary)                                             as salary_amount,
       (sum(manager_sales) - (sum(product_sum) + sum(manager_salary))) as profit_amount
from (
         select extract(MONTH from ORDER_DATE)            month,
                sum(SALE_AMOUNT)                       as manager_sales,  -- продажи менеджера по месяцу
                30000 + 0.05 * sum(SALE_AMOUNT)        as manager_salary, -- зп менеджера в месяце
                sum(PRODUCT_PRICE * 0.9 * PRODUCT_QTY) as product_sum     -- исходная стоимость товаров проданных менеджером в месяце
         from V_FACT_SALE
         where ORDER_DATE between '01.01.2014' and '31.12.2014'
         group by extract(MONTH from ORDER_DATE), MANAGER_ID
     )
group by month
order by month;
