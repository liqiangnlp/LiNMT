use Encode;
use utf8;

if (scalar(@ARGV) ne 3)
{
  print STDERR "perl LiNMT-num.pl config input output\n";
  exit(1);
}



$config_string = "";
open (CONFIG, "<:encoding(utf8)", $ARGV[0]) or die "Error: can not read $ARGV[0]!\n";
while (my $line = <CONFIG>)
{
  $line =~ s/[\r\n]//g;
  $config_string .= $line;
}
close(CONFIG);



open (INPUT, "<:encoding(utf8)", $ARGV[1]) or die "Error: can not read $ARGV[1]!\n";
open (OUTPUT, ">:encoding(utf8)", $ARGV[2]) or die "Error: can not write $ARGV[2]!\n";
$line_num = 0;
$find_num = 0;
while (my $line = <INPUT>)
{
  ++$line_num;
  $line =~ s/[\r\n]//g;
  if ($line =~ /[[0-9$config_string]/)
  {
    ++$find_num;
    print OUTPUT $line."\n";
    print STDERR "\rLINE=$line_num FIND=$find_num";
  }
}
print STDERR "\rLINE=$line_num FIND=$find_num\n";


close(INPUT);
close(OUTPUT);

