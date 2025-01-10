# method="chirp optionRL catrl optioncritic drl"
# domains="maze_continuous fourrooms_continuous taxi_pass1_continuous office_continuous minecraft_continuous"
methods="chirp"
domains="taxi_pass1_continuous"
trials="1 2 3 4 5 6 7 8 9 10"

# Runs all the methods, domains, and trials on different threads

for method in $methods; do
    for domain in $domains; do
        for trial in $trials; do
            echo $method
            echo $domain
            echo $trial
            log_path=logs/"$domain"_"$method"_trial_"$trial".log
            echo $log_path
            
            if [ "$method" = chirp ]; then
            python3 optionCATs.py "$method" "$domain" "$trial" > "$log_path" &
            fi 

            if [ "$method" = catrl ]; then
            python baseline_catrl.py "$method" "$domain" "$trial" > "$log_path" &
            fi 

            if [ "$method" = optioncritic ]; then
            python baseline_optioncritic.py "$method" "$domain" "$trial" > "$log_path" &
            fi 

            if [ "$method" = drl ]; then
            python baseline_drl.py "$method" "$domain" "$trial" > "$log_path" &
            fi 
        done
    done
done

