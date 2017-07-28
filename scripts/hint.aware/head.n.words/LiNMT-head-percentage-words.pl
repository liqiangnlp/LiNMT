#!/usr/bin/perl

use POSIX;

# * LiNMT v1.0                          * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 07/14/2017                  * #
# * Time  : 15:31                       * #
# * Head % words in a sentences         * #


if (scalar(@ARGV) ne 3) {
  print STDERR "perl LiNMT-head-percentage-words.pl PERCENT INPUT OUTPUT\n";
  exit(1);
}


my $percentage = $ARGV[0];
open (INPUT, "<", $ARGV[1]) or die "Error: can not open $ARGV[1]!\n";
open (OUTPUT, ">", $ARGV[2]) or die "Error: can not write $ARGV[2]!\n";
open (LOG, ">", $ARGV[2].".log") or die "Error: can not write $ARGV[2].log!\n";

my $line_no = 0;
my $total_words = 0;
my $reserved_words = 0;
while (my $sent = <INPUT>)
{
  ++$line_no;

  $sent =~ s/[\r\n]//g;
  $sent =~ s/\s+$//g;
  $sent =~ s/^\s+//g;
  
  my @domains = split / \|\|\|\| /, $sent;

  my @words = split /\s+/, $domains[0];
  my $words_number = scalar(@words);
  $total_words += $words_number;
  my $reserved_number_float = $words_number * $percentage;
  my $reserved_number = ceil($reserved_number_float);
  
  my $output_sentence = "";
  for my $i (0..$words_number - 1)
  {
    if ($i < $reserved_number)
	{
	  $output_sentence .= " ".$words[$i];
	  ++$reserved_words;
	}
	else
	{
	  $output_sentence .= " #";
	}
  }
  $output_sentence =~ s/^\s+//g;
  $output_sentence =~ s/\s+$//g;
  if (scalar(@domains) eq 2)
  {
    $output_sentence .= " |||| ".$domains[1];
  }
  
  print OUTPUT $output_sentence."\n";

  if ($line_no % 100 == 0) 
  {
    $percentage_reserved_words = $reserved_words / $total_words;
    print STDERR "\r$line_no  TOTAL=$total_words  RESERVED=$reserved_words  PERT=$percentage_reserved_words";
  }
}

$percentage_reserved_words = $reserved_words / $total_words;
print STDERR "\r$line_no  TOTAL=$total_words  RESERVED=$reserved_words  PERT=$percentage_reserved_words\n";
print LOG "LINE=$line_no  TOTAL=$total_words  RESERVED=$reserved_words  PERT=$percentage_reserved_words\n";

close (INPUT);
close (OUTPUT);
close (LOG);




