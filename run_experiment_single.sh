# method=$1 
# domain=$2 
# trial=$3
# method="chirp catrl optioncritic drl"
# domain="maze_continuous" "fourrooms_continuous" "taxi_pass1_continuous" "office_continuous" "minecraft_continuous"
method="chirp"
domain="fourrooms_continuous"
trial="2"

echo $method
echo $domain
echo $trial

log_path=logs/"$domain"_"$method"_trial_"$trial".log
echo $log_path

# Runs single method for one domain and trial


if [ "$method" = chirp ]; then
python3 optionCATs.py "$method" "$domain" "$trial" 
# > "$log_path" 
fi 

if [ "$method" = catrl ]; then
python baseline_catrl.py "$method" "$domain" "$trial" 
# > "$log_path"
fi 

if [ "$method" = optioncritic ]; then
python baseline_optioncritic.py "$method" "$domain" "$trial" 
# > "$log_path"
fi 

if [ "$method" = drl ]; then
python baseline_drl.py "$method" "$domain" "$trial" 
# > "$log_path"
fi 

